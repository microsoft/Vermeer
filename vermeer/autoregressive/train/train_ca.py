# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
#   train_c2i.py
"""
Training script for Channel-Adaptive GPT (gpt_ca) models.

Trains on CA image codes with SOC/EOC markers for multi-channel generation.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from glob import glob
from copy import deepcopy
import os
import time
import inspect
import argparse

import collections
import wandb
import math
import random
import numpy as np

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.build import build_dataset
from autoregressive.models.gpt_ca import GPT_models
from dataset.ca_image import build_ca_code
from utils.lr_control import lr_wd_annealing
from utils.optim import creat_optimizer, creat_llrd_optimizer
# from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR


#################################################################################
#                             Training Helper Functions                         #
#################################################################################
# def custom_collate_fn(batch):
#     """
#     Custom collate function for CA dataset.
    
#     Each item is (features_with_eos, labels, n_channels).
#     We need to handle n_channels separately since it's the same for all items.
#     """
#     features = torch.stack([item[0] for item in batch])
#     labels = torch.stack([item[1] for item in batch])
#     n_channels = batch[0][2]  # Same for all items in batch
#     return features, labels, n_channels

def apply_channel_augmentation(features_batch, n_channels, min_channels, tokens_per_channel=258):
    """
    Apply channel reordering/dropout augmentation to a batch of token sequences.

    Randomly selects a subset and ordering of channels. The same selection applies
    to all samples in the batch to keep sequence lengths uniform.

    SOC/EOC tokens keep their original channel identity (e.g., SOC_2 stays SOC_2
    regardless of position in the sequence).

    Args:
        features_batch: (B, seq_len) tensor of token sequences.
            Structure: [SOC_0, codes_0, EOC_0, ..., SOC_{n-1}, codes_{n-1}, EOC_{n-1}, EOS]
        n_channels: int, number of channels in the original sequence
        min_channels: int, minimum number of channels to keep
        tokens_per_channel: int, tokens per channel block (SOC + codes + EOC), default 258

    Returns:
        (augmented_features, num_keep): augmented tensor and number of channels kept
    """
    num_keep = random.randint(min_channels, n_channels)
    selected_channels = random.sample(range(n_channels), num_keep)

    batch_size = features_batch.shape[0]
    # Last token is EOS
    eos_token = features_batch[:, -1:]  # (B, 1)

    # Split into channel blocks (excluding EOS)
    channel_blocks = []
    for ch_idx in range(n_channels):
        start = ch_idx * tokens_per_channel
        end = start + tokens_per_channel
        channel_blocks.append(features_batch[:, start:end])  # (B, tokens_per_channel)

    # Select and reorder blocks
    selected_blocks = [channel_blocks[ch] for ch in selected_channels]

    # Concatenate selected blocks + EOS
    augmented = torch.cat(selected_blocks + [eos_token], dim=1)

    return augmented, num_keep


def apply_channel_config(features_batch, n_channels, channel_config, tokens_per_channel=258):
    """
    Select and reorder channel blocks according to a fixed channel_config.

    Args:
        features_batch: (B, seq_len) tensor with structure [SOC_0, codes_0, EOC_0, ..., EOS]
        n_channels: int, original number of channels
        channel_config: list[int], channel indices to select (in desired order)
        tokens_per_channel: int, default 258
    Returns:
        (selected_features, len(channel_config))
    """
    if channel_config is None or len(channel_config) == 0:
        raise ValueError("channel_config must contain at least one channel index")
    invalid_channels = [ch for ch in channel_config if ch < 0 or ch >= n_channels]
    if invalid_channels:
        raise ValueError(
            f"channel_config contains out-of-range channel indices: {invalid_channels}. "
            f"Valid range is [0, {n_channels - 1}] for n_channels={n_channels}."
        )

    eos_token = features_batch[:, -1:]
    channel_blocks = []
    for ch_idx in range(n_channels):
        start = ch_idx * tokens_per_channel
        end = start + tokens_per_channel
        channel_blocks.append(features_batch[:, start:end])
    selected_blocks = [channel_blocks[ch] for ch in channel_config]
    return torch.cat(selected_blocks + [eos_token], dim=1), len(channel_config)


def make_collate_fn(channel_augment=False, min_channels=2, channel_augment_prob=0.5,
                    n_channels=4, tokens_per_channel=258, channel_config=None):
    """
    Factory function that returns a collate callable with augmentation config captured.

    Args:
        channel_augment: bool, whether to enable channel reordering/dropout
        min_channels: int, minimum channels to keep when augmenting
        channel_augment_prob: float, probability of applying augmentation per batch
        n_channels: int, number of channels in the dataset
        tokens_per_channel: int, tokens per channel block (SOC + codes + EOC)
    """
    def collate_fn(batch):
        """
        Custom collate function for CA dataset with optional channel augmentation.

        Each item is (features_with_eos, labels, n_channels).

        Returns
        features: (B, seq_len) # features for the CA images
        labels: (B, seq_len) # labels for the CA images
        n_channels_out: int # number of channels (possibly reduced by augmentation)
        padding_mask: (B, seq_len) # True for real tokens, False for padding
        lens: (B,) # length of each sequence
        """
        features = torch.stack([item[0] for item in batch])
        labels_list = [item[1] for item in batch]
        n_channels_batch = batch[0][2]  # Currently assuming this is the same for all items in batch
        if n_channels is not None and n_channels_batch != n_channels:
            raise ValueError(
                f"Batch n_channels ({n_channels_batch}) does not match expected n_channels ({n_channels}). "
                "Check dataset configuration and training arguments."
            )

        # Apply deterministic channel config if set (before channel augmentation)
        if channel_config is not None:
            features, n_channels_batch = apply_channel_config(
                features, n_channels_batch, channel_config, tokens_per_channel
            )

        # Apply channel augmentation if enabled
        if channel_augment and random.random() < channel_augment_prob:
            features, n_channels_batch = apply_channel_augmentation(
                features, n_channels_batch, min_channels, tokens_per_channel
            )

        # Check if labels have variable shapes (ca_esm_embed_full case)
        if len(labels_list[0].shape) > 1:  # Multi-dimensional labels (e.g., (#AA, embed_dim))
            # Find max sequence length in batch
            max_len = max(label.shape[0] for label in labels_list)
            embed_dim = labels_list[0].shape[1]
            batch_size = len(labels_list)

            # Create padding mask: True for real tokens, False for padding
            padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

            # Pad all labels to max_len with zeros
            padded_labels = []
            lens = []
            for i, label in enumerate(labels_list):
                seq_len = label.shape[0]
                lens.append(seq_len)
                # Mark real tokens as True in the mask
                padding_mask[i, :seq_len] = True

                if seq_len < max_len:
                    # Pad with zeros
                    padding = torch.zeros(max_len - seq_len, embed_dim, dtype=label.dtype)
                    padded_label = torch.cat([label, padding], dim=0)
                else:
                    padded_label = label
                padded_labels.append(padded_label)

            labels = torch.stack(padded_labels)
            lens = torch.tensor(lens)

            return features, labels, n_channels_batch, padding_mask, lens
        else:
            # For ca, ca_binary_prefix, or ca_esm_embed_mean_pool: labels are already same shape (e.g., scalar or fixed-length)
            labels = torch.stack(labels_list)

            return features, labels, n_channels_batch, None, None

    return collate_fn

    
   

#################################################################################
#                         Training/Validation Functions                         #
#################################################################################

def eval_ep(model, val_loader, val_loss, val_steps, device, args, ptdtype):
    with torch.no_grad():
        for z_with_eos, y, n_channels, np_mask, lens in val_loader:
            z_with_eos = z_with_eos.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if np_mask is not None:
                np_mask = np_mask.to(device, non_blocking=True)
                lens = lens.to(device, non_blocking=True)
            # Input: all tokens except last
            z_indices = z_with_eos[:, :-1]
            # Targets: all tokens including last token (EOS)
            targets = z_with_eos

            if args.gpt_type == 'ca':
                # cond idx = 0s of shape (B, 1)
                with torch.amp.autocast('cuda', dtype=ptdtype):
                    _, loss = model(
                        idx=z_indices,
                        cond_idx=torch.zeros((z_indices.shape[0], 1), device=device, dtype=torch.long), # TODO: is torch.long correct? vs ptdtype?
                        targets=targets
                    )
            elif args.gpt_type == 'ca_binary_prefix':
                with torch.amp.autocast('cuda', dtype=ptdtype):
                    _, loss = model(
                        idx=z_indices,
                        cond_idx=y,
                        targets=targets
                    )
            elif args.gpt_type == 'ca_esm_embed_mean_pool':
                y = y.unsqueeze(1) # TODO: check this shape 

                with torch.amp.autocast('cuda', dtype=ptdtype):
                    _, loss = model(
                        idx=z_indices,
                        cond_idx=y,
                        targets=targets
                    )
            elif args.gpt_type == 'ca_esm_embed_full':
                with torch.amp.autocast('cuda', dtype=ptdtype):
                    _, loss = model(
                        idx=z_indices,
                        cond_idx=y,
                        targets=targets,
                        non_mask=np_mask,
                        lens=lens
                    )
            
            val_loss += loss.item()
            val_steps += 1

    return val_loss, val_steps




#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    random.seed(seed) 
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace("/", "-")  # e.g., GPT-XL/2 --> GPT-XL-2 (for naming folders)
        
        # Build experiment name with optional experiment_name prefix
        if args.experiment_name:
            experiment_folder_name = f"{experiment_index:03d}-{args.experiment_name}-{model_string_name}"
        else:
            experiment_folder_name = f"{experiment_index:03d}-{model_string_name}"
        
        experiment_dir = f"{args.results_dir}/{experiment_folder_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_folder_name}/checkpoints"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
        
        # Initialize wandb
        if args.use_wandb:
            wandb_run_name = f"{experiment_index:03d}-{args.experiment_name}-{model_string_name}" if args.experiment_name else f"{experiment_index:03d}-{model_string_name}"
            wandb.init(
                project=args.wandb_project,
                name=wandb_run_name,
                config=vars(args),
                dir=experiment_dir,
                resume="allow",
                id=args.wandb_run_id if args.wandb_run_id else None
            )
            logger.info(f"Weights & Biases initialized for project: {args.wandb_project}")
    
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Parse channel_config while preserving original args.n_channels
    channel_config = None
    n_max_channels_for_model = args.n_channels  # preserve original for dataset + model embedding table
    selected_n_channels = args.n_channels
    if args.channel_config is not None:
        channel_config = [int(x) for x in args.channel_config.split(',')]
        assert all(0 <= c < args.n_channels for c in channel_config), \
            f"channel_config indices {channel_config} must be in [0, n_channels={args.n_channels})"
        selected_n_channels = len(channel_config)
        logger.info(
            f"channel_config={channel_config} → using {selected_n_channels} selected channels "
            f"(dataset/model n_max_channels={n_max_channels_for_model})"
        )

    # Setup model
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    
    latent_size = args.image_size // args.downsample_size
    block_size_per_channel = latent_size ** 2  # e.g., 256 for 256x256 images with 16x downsampling
    
    print("args.gpt_ckpt:", args.gpt_ckpt)
    print("args.pretrained_gpt_ckpt", args.pretrained_gpt_ckpt)
    assert not (args.gpt_ckpt is not None and args.pretrained_gpt_ckpt is not None), "only either one of gpt_ckpt or pretrained_gpt_ckpt can be provided"
    if args.pretrained_gpt_ckpt:
        pretrained_gpt = True
    else:
        pretrained_gpt = False
    
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size_per_channel=block_size_per_channel,
        n_max_channels=n_max_channels_for_model,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        pretrained_gpt=pretrained_gpt,
    ).to(device)
    logger.info(f"GPT-CA Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Model config: n_max_channels={n_max_channels_for_model}, block_size_per_channel={block_size_per_channel}")
    logger.info(f"Extended vocab size: {model.extended_vocab_size}")

    if args.ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    # Setup optimizer
    if args.llrd == -1: # no layererwise learning rate decay
        # Use standard optimizer without LLRD
        optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)
    else:
        # args.llrd must be greater than 0.0 and leq than 1.0 if specified
        assert args.llrd > 0.0 and args.llrd < 1.0, "args.llrd must be greater than 0.0 and less than 1.0"
        # Use layer-wise learning rate decay
        optimizer = creat_llrd_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger, args.llrd)

    # Validate channel augmentation args
    effective_n_channels_for_training = selected_n_channels if channel_config is not None else args.n_channels
    if args.channel_augment:
        assert 1 <= args.min_channels <= effective_n_channels_for_training, \
            (
                f"min_channels ({args.min_channels}) must be between 1 and effective n_channels "
                f"({effective_n_channels_for_training})."
            )

    # Setup data:
    tokens_per_channel = (args.image_size // args.downsample_size) ** 2 + 2  # e.g., 258 for 256x256 with 16x downsample
    train_dataset = build_ca_code(args, split='train')
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    collate_fn_train = make_collate_fn(
        channel_augment=args.channel_augment,
        min_channels=args.min_channels,
        channel_augment_prob=args.channel_augment_prob,
        n_channels=args.n_channels,
        tokens_per_channel=tokens_per_channel,
        channel_config=channel_config,
    )
    collate_fn_val = make_collate_fn(
        channel_augment=False,
        n_channels=args.n_channels,
        tokens_per_channel=tokens_per_channel,
        channel_config=channel_config,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_train
    )
    logger.info(f"Train Dataset contains {len(train_dataset):,} CA images ({args.code_path})")
    if args.channel_augment:
        logger.info(
            f"Channel augmentation enabled: min_channels={args.min_channels}, prob={args.channel_augment_prob}, "
            f"effective_n_channels={effective_n_channels_for_training}"
        )

    # Setup validation data:
    val_dirs = args.val_dirs.split(',')
    val_loaders = {}
    for val_dir in val_dirs:
        val_dataset = build_ca_code(args, split=val_dir)
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
            seed=args.global_seed
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.global_batch_size // dist.get_world_size()),
            shuffle=False,
            sampler=val_sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True, # set True to make compatible with torch.compile, alternatively set torch.compile(*,dynamic=True) at some expense of performance
            collate_fn=collate_fn_val
        )
        val_loaders[val_dir] = val_loader
        logger.info(f"Validation dataset {val_dir} contains {len(val_dataset):,} CA images")

    # Prepare models for training:
    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
        if args.ema:
            ema.load_state_dict(checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"]) if "optimizer" in checkpoint else None 
        train_steps = checkpoint["steps"] if "steps" in checkpoint else 0
        start_epoch = int(train_steps / int(len(train_dataset) / args.global_batch_size))
        train_steps = int(start_epoch * int(len(train_dataset) / args.global_batch_size))
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    elif args.pretrained_gpt_ckpt:
        try:
            checkpoint = torch.load(args.pretrained_gpt_ckpt, map_location="cpu",)
        except:
            # if using microscoppy-trained model as pre-trained gpt ckpt, i.e. in fine-tuning after channel-augmented pre-training
            checkpoint = torch.load(args.pretrained_gpt_ckpt, map_location="cpu", weights_only=False) 

        # pretrained_state = checkpoint["model"] # logic for B, L, XL models 
        if "model" in checkpoint:
            pretrained_state = checkpoint["model"] # logic for B, L, XL models 
        else:
            pretrained_state = checkpoint # logic for XXL model
        # Filter out keys that shouldn't be loaded from pretrained checkpoint

        keys_to_skip = [
            "cls_embedding.embedding_table.weight",  # new cls embeddings
        ]
        filtered_state = {k: v for k, v in pretrained_state.items() if k not in keys_to_skip}
        
        model.load_state_dict(filtered_state, strict=False) # load in pretrained gpt weights  
        model.extend_head() # copy weights from pretrained gpt to extended head
        model.extend_tok_embeddings() # copy weights from pretrained gpt to extended tok_embeddings
        del checkpoint
        logger.info(f"Loaded pretrained GPT checkpoint: {args.pretrained_gpt_ckpt}")
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    # if args.channel_augment and not args.no_compile:
    #     args.no_compile = True
    #     logger.info("Disabling torch.compile due to channel augmentation (variable sequence lengths)")

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        if args.channel_augment:
            logger.info("compiling the model with dynamic=True due to channel augmentation (variable sequence lengths)")
            model = torch.compile(model, dynamic=True) # requires PyTorch 2.0
        else:
            logger.info("compiling the model with dynamic=False")
            model = torch.compile(model) # requires PyTorch 2.0
    
    model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler('cuda',enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    iters_train = len(train_loader)
    max_it = args.epochs * iters_train
    wp_it = args.warmup_epochs * iters_train

    print("wp_it:", wp_it)
    print("iters_train:", iters_train)
    print("max_it:", max_it)
    print("args.wp0:", args.wp0)
    print("args.wpe:", args.wpe)
    print("args.lr:", args.lr)
    print("args.schedule:", args.lr_schedule)

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for it, (z_with_eos, y, n_channels, np_mask, lens) in enumerate(train_loader): # dataloader adds EOS token
            g_it = epoch * iters_train + it 

            # print("g_it:", g_it)
            if args.lr_schedule is not None:
                min_tlr, max_tlr = lr_wd_annealing(args.lr_schedule, optimizer, peak_lr=args.lr, cur_it=g_it, wp_it=wp_it, max_it=max_it, wp0=args.wp0, wpe=args.wpe)
            else:
                min_tlr, max_tlr = args.lr, args.lr

            # print("min_tlr:", min_tlr)
            # print("max_tlr:", max_tlr)

            z_with_eos = z_with_eos.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True) ## todo: inefficient for ca case 
            
            # Input: all tokens except last
            z_indices = z_with_eos[:, :-1]  # (B, seq_len-1)
            # Targets: all tokens including last token (EOS)
            targets = z_with_eos  # (B, seq_len)

            if args.gpt_type == 'ca':
                # cond idx = 0s of shape (B, 1)
                with torch.amp.autocast('cuda', dtype=ptdtype):
                    _, loss = model(
                        idx=z_indices,
                        cond_idx=torch.zeros((z_indices.shape[0], 1), device=device, dtype=torch.long), # TODO: is torch.long correct? vs ptdtype?
                        targets=targets
                    )
            elif args.gpt_type == 'ca_binary_prefix':
                with torch.amp.autocast('cuda', dtype=ptdtype):
                    _, loss = model(
                        idx=z_indices,
                        cond_idx=y,
                        targets=targets
                    )
            elif args.gpt_type == 'ca_esm_embed_mean_pool':
                y = y.unsqueeze(1) # TODO: check this shape 

                # print("y shape:", y.shape)
                # print("z_indices shape:", z_indices.shape)
                # print("targets shape:", targets.shape)

                with torch.amp.autocast('cuda', dtype=ptdtype):
                    _, loss = model(
                        idx=z_indices,
                        cond_idx=y,
                        targets=targets
                    ) 
            elif args.gpt_type == 'ca_esm_embed_full': # TODO: double check this
                # print("Y shape:", y.shape)
                # print("z_indices shape:", z_indices.shape)
                # print("targets shape:", targets.shape)
                with torch.amp.autocast('cuda', dtype=ptdtype):
                    _, loss = model(
                        idx=z_indices,
                        cond_idx=y,
                        targets=targets,
                        non_mask=np_mask,
                        lens=lens
                    )
            else:
                raise ValueError(f"Unsupported model type: {args.gpt_type}")
            # TODO: add support for esm_embed_full and localization_onehot, deprecate binary prefix
            
            # backward pass, with gradient scaling if training in fp16         
            scaler.scale(loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            if args.ema:
                update_ema(ema, model.module._orig_mod if not args.no_compile else model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                
                # Log to wandb
                if rank == 0 and args.use_wandb:
                    log_dict = {
                        "train/loss": avg_loss,
                        "train/log_loss": np.log(avg_loss),
                        "train/steps_per_sec": steps_per_sec,
                        "train/epoch": epoch,
                        "train/step": train_steps,
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                    }
                    wandb.log(log_dict, step=train_steps)
                
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Validation:
            if train_steps % args.val_every == 0 and train_steps > 0:
                model.eval()

                val_losses = {}
                val_steps_dict = {}
                for val_dir in val_dirs:
                    val_losses[val_dir] = 0.0
                    val_steps_dict[val_dir] = 0

                avg_val_losses = {}

                logger.info(f"Starting validation at step {train_steps}...")

                for val_dir in val_dirs:
                    val_loss, val_steps = eval_ep(model=model, val_loader=val_loaders[val_dir], val_loss=val_losses[val_dir], val_steps=val_steps_dict[val_dir], device=device, args=args, ptdtype=ptdtype)

                    if val_steps > 0:
                        # Aggregate validation loss
                        val_loss_tensor = torch.tensor(val_loss / val_steps, device=device)
                        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                        avg_val_loss = val_loss_tensor.item() / dist.get_world_size()
                        avg_val_losses[val_dir] = avg_val_loss
                        logger.info(f"(step={train_steps:07d}) Val Loss {val_dir}: {avg_val_loss:.4f}")

                if rank == 0 and args.use_wandb:
                    val_log_dict = {"val/step": train_steps}
                    for val_dir in val_dirs:
                        if val_dir in avg_val_losses:
                            val_log_dict[f"val/{val_dir}/loss"] = avg_val_losses[val_dir]
                            val_log_dict[f"val/{val_dir}/log_loss"] = np.log(avg_val_losses[val_dir])

                        wandb.log(val_log_dict, step=train_steps)
                
                model.train()
                torch.cuda.empty_cache() # release memory 

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if not args.no_compile:
                        model_weight = model.module._orig_mod.state_dict()
                    else:
                        model_weight = model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, cloud_checkpoint_path)
                    logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    
    # Finish wandb run
    if rank == 0 and args.use_wandb:
        wandb.finish()
    
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--cloud-save-path", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--experiment-name", type=str, default=None, help='experiment name to prepend to checkpoint directory')
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['ca', 'ca_binary_prefix', 'ca_esm_embed_mean_pool', 'ca_esm_embed_full'], default="ca", help="type of conditioning")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--pretrained-gpt-ckpt", type=str, default=None, help="pretrained GPT model checkpoint path")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='ca_code')
    parser.add_argument("--val-dirs", type=str, default='val', help="comma-separated list of directories to use for validation")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--n-channels", type=int, default=4, help="number of channels in CA images")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", type=str, choices=['cos', 'lin', 'lin0', 'lin00', 'linX', 'exp'], default=None, help="Learning rate schedule")
    parser.add_argument("--warmup-epochs", type=int, default=30, help="Number of warmup epochs")
    parser.add_argument("--wp0", type=float, default=0.005, help="Learning rate at the start of warmup (fraction of peak_lr)")
    parser.add_argument("--wpe", type=float, default=0.01, help="Minimum learning rate at end of training (fraction of peak_lr)")
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--label-smooth", type=float, default=0.0, help="Label smoothing factor")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--val-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--llrd", type=float, default=-1, help="Layer-wise learning rate decay factor (-1 means disabled, 0.9 recommended for fine-tuning)")
    parser.add_argument("--use-wandb", action='store_true', help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="llamagen-ca", help="W&B project name")
    parser.add_argument("--wandb-run-id", type=str, default=None, help="W&B run ID for resuming")
    parser.add_argument("--channel-augment", action='store_true', help="Enable channel reordering/dropout augmentation")
    parser.add_argument("--min-channels", type=int, default=2, help="Minimum channels to keep when channel augmentation is enabled (default: 2)")
    parser.add_argument("--channel-augment-prob", type=float, default=0.5, help="Probability of applying channel augmentation per batch (default: 0.5)")
    parser.add_argument("--channel-config", type=str, default=None,
        help="Comma-separated channel indices to select and order, e.g. '0,1,3' or '3,0'. Default: use all channels in original order.")
    args = parser.parse_args()
    main(args)

