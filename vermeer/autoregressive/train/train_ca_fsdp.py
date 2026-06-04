# Modified from:
#   train_ca.py (DDP version)
#   train_c2i_fsdp.py (FSDP patterns)
"""
FSDP training script for Channel-Adaptive GPT (gpt_ca) models.

Uses Fully Sharded Data Parallel for training large models (GPT-XXL, GPT-3B)
that don't fit in single-GPU memory with DDP. Key differences from train_ca.py:
  - No EMA (complex with FSDP)
  - No LLRD (FSDP flattens params, breaking dimension-based grouping)
  - No torch.compile (compatibility issues with FSDP)
  - Name-based optimizer (decay/no-decay by param name)
  - FSDP checkpointing: consolidated.pth + per-rank optimizer shards
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

import os
import time
import inspect
import functools
import argparse
import contextlib
from glob import glob

import wandb
import numpy as np

from utils.logger import create_logger
from autoregressive.models.gpt_ca import GPT_models
from dataset.ca_image import build_ca_code
from utils.lr_control import lr_wd_annealing


#################################################################################
#                             FSDP Setup Functions                              #
#################################################################################

def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace, device) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        device_id=device,
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
            "hsdp": ShardingStrategy.HYBRID_SHARD,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.mixed_precision],
            reduce_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.grad_precision or args.mixed_precision],
        ),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )

    torch.cuda.synchronize()

    return model


def creat_optimizer_by_name(model, weight_decay, learning_rate, betas, global_rank, logger, use_fused=True):
    # start with all of the candidate parameters
    all_param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in all_param_dict.items() if p.requires_grad}

    # create optim groups.
    # Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.

    # FSDP flattens parameters, so we use name-based grouping:
    # all 'norm' params get no weight decay, everything else decays
    decay_params = [p for n, p in param_dict.items() if 'norm' not in n]
    nodecay_params = [p for n, p in param_dict.items() if 'norm' in n]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"(rank {global_rank}) num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"(rank {global_rank}) num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    # Fused Adam is incompatible with FSDP checkpoint resume (dtype/device mismatch)
    fused_available = use_fused and 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer


def _disable_fused_adam(optimizer: torch.optim.Optimizer) -> bool:
    """
    Ensure resumed AdamW does not use fused backend.

    load_state_dict() restores param group hyperparameters from checkpoint.
    If the checkpoint was saved with fused=True, that value overrides the
    constructor argument and can trigger dtype/layout checks in fused Adam.
    """
    changed = False
    if "fused" in optimizer.defaults and optimizer.defaults["fused"]:
        optimizer.defaults["fused"] = False
        changed = True
    for group in optimizer.param_groups:
        if group.get("fused", False):
            group["fused"] = False
            changed = True
    return changed


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def custom_collate_fn(batch):
    """
    Custom collate function for CA dataset with ESM full conditioning.

    Each item is (features_with_eos, labels, n_channels).

    For ca_esm_embed_full, labels have variable shapes (#AA, embed_dim), so we pad them.
    Returns
    features: (B, seq_len)
    labels: (B, seq_len) or (B, max_aa_len, embed_dim)
    n_channels: int
    padding_mask: (B, max_aa_len) or None
    lens: (B,) or None
    """
    features = torch.stack([item[0] for item in batch])
    labels_list = [item[1] for item in batch]
    n_channels = batch[0][2]

    # Check if labels have variable shapes (ca_esm_embed_full case)
    if len(labels_list[0].shape) > 1:  # Multi-dimensional labels
        max_len = max(label.shape[0] for label in labels_list)
        embed_dim = labels_list[0].shape[1]
        batch_size = len(labels_list)

        padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        padded_labels = []
        lens = []
        for i, label in enumerate(labels_list):
            seq_len = label.shape[0]
            lens.append(seq_len)
            padding_mask[i, :seq_len] = True
            if seq_len < max_len:
                padding = torch.zeros(max_len - seq_len, embed_dim, dtype=label.dtype)
                padded_label = torch.cat([label, padding], dim=0)
            else:
                padded_label = label
            padded_labels.append(padded_label)

        labels = torch.stack(padded_labels)
        lens = torch.tensor(lens)
        return features, labels, n_channels, padding_mask, lens
    else:
        labels = torch.stack(labels_list)
        return features, labels, n_channels, None, None


#################################################################################
#                         Validation Function                                   #
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
                with torch.cuda.amp.autocast(dtype=ptdtype) if ptdtype != torch.float32 else contextlib.nullcontext():
                    _, loss = model(
                        idx=z_indices,
                        cond_idx=torch.zeros((z_indices.shape[0], 1), device=device, dtype=torch.long),
                        targets=targets
                    )
            elif args.gpt_type == 'ca_binary_prefix':
                with torch.cuda.amp.autocast(dtype=ptdtype) if ptdtype != torch.float32 else contextlib.nullcontext():
                    _, loss = model(
                        idx=z_indices,
                        cond_idx=y,
                        targets=targets
                    )
            elif args.gpt_type == 'ca_esm_embed_mean_pool':
                y = y.unsqueeze(1)
                with torch.cuda.amp.autocast(dtype=ptdtype) if ptdtype != torch.float32 else contextlib.nullcontext():
                    _, loss = model(
                        idx=z_indices,
                        cond_idx=y,
                        targets=targets
                    )
            elif args.gpt_type == 'ca_esm_embed_full':
                with torch.cuda.amp.autocast(dtype=ptdtype) if ptdtype != torch.float32 else contextlib.nullcontext():
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
    assert not args.ema, "EMA is not supported with FSDP. Remove --ema flag."

    # =======================================
    #    Initialize Distributed Training
    # =======================================
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    global_rank = dist.get_rank()
    device = global_rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + global_rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={global_rank}, device={device}, seed={seed}, world_size={dist.get_world_size()}.")

    # =======================================
    #    Initialize logger and wandb
    # =======================================
    timestamp = None
    if global_rank == 0:
        timestamp = time.localtime()
        timestamp = int(time.strftime("%Y%m%d%H%M%S", timestamp))
    # Broadcast timestamp to all processes
    timestamp_tensor = torch.tensor([timestamp] if timestamp is not None else [0.0], dtype=torch.double).to(device)
    dist.broadcast(timestamp_tensor, src=0)
    timestamp = int(timestamp_tensor.item())

    model_string_name = args.gpt_model.replace("/", "-")
    if args.experiment_name:
        experiment_folder_name = f"{timestamp}-{args.experiment_name}-{model_string_name}"
    else:
        experiment_folder_name = f"{timestamp}-{model_string_name}"

    experiment_dir = f"{args.results_dir}/{experiment_folder_name}"
    cloud_checkpoint_dir = f"{args.cloud_save_path}/{experiment_folder_name}"

    if global_rank == 0:
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)

        # Initialize wandb
        if args.use_wandb:
            os.environ["WANDB_DIR"] = experiment_dir
            wandb.init(
                project=args.wandb_project,
                name=experiment_folder_name,
                config=vars(args),
                resume="allow",
                id=args.wandb_run_id if args.wandb_run_id else None
            )
            logger.info(f"Weights & Biases initialized for project: {args.wandb_project}")
    else:
        logger = create_logger(None)

    logger.info(f"Experiment directory created at {experiment_dir}")
    logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    logger.info(f"{args}")
    logger.info(f"Starting rank={global_rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # ======================================================
    #     Initialize model and load weights
    # ======================================================
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p

    latent_size = args.image_size // args.downsample_size
    block_size_per_channel = latent_size ** 2

    assert not (args.gpt_resume is not None and args.pretrained_gpt_ckpt is not None), \
        "only one of gpt_resume or pretrained_gpt_ckpt can be provided"

    if args.pretrained_gpt_ckpt:
        pretrained_gpt = True
    else:
        pretrained_gpt = False

    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size_per_channel=block_size_per_channel,
        n_max_channels=args.n_channels,
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
    logger.info(f"Model config: n_max_channels={args.n_channels}, block_size_per_channel={block_size_per_channel}")
    logger.info(f"Extended vocab size: {model.extended_vocab_size}")

    # Load weights on rank 0 BEFORE FSDP wrapping (sync_module_states broadcasts to others)
    if args.gpt_resume:
        if global_rank == 0:
            logger.info(f"Resuming model weights from: {args.gpt_resume}")
            model.load_state_dict(torch.load(os.path.join(
                args.gpt_resume, "consolidated.pth",
            ), map_location="cpu"), strict=False)
    elif args.pretrained_gpt_ckpt:
        if global_rank == 0:
            checkpoint = torch.load(args.pretrained_gpt_ckpt, map_location="cpu")
            # pretrained_state = checkpoint["model"] # logic for B, L, XL models 
            if "model" in checkpoint:
                pretrained_state = checkpoint["model"] # logic for B, L, XL models 
            else:
                pretrained_state = checkpoint # logic for XXL model
            # Filter out keys that shouldn't be loaded from pretrained checkpoint
            keys_to_skip = [
                "cls_embedding.embedding_table.weight",
            ]
            filtered_state = {k: v for k, v in pretrained_state.items() if k not in keys_to_skip}
            model.load_state_dict(filtered_state, strict=False)
            model.extend_head()
            model.extend_tok_embeddings()
            del checkpoint
            logger.info(f"Loaded pretrained GPT checkpoint: {args.pretrained_gpt_ckpt}")

    # Wrap model with FSDP
    model = setup_fsdp_sync(model, args, device)

    # ======================================================
    #     Initialize optimizer and resume
    # ======================================================
    # Disable fused Adam when resuming: fused Adam requires params/grads/states
    # to share the same dtype and device, which breaks when loading optimizer
    # shards saved in fp32 into an FSDP model with bf16 mixed precision.
    use_fused = not args.gpt_resume
    optimizer = creat_optimizer_by_name(model, args.weight_decay, args.lr, (args.beta1, args.beta2), global_rank, logger, use_fused=use_fused)

    if args.gpt_resume:
        opt_state_world_size = len([
            x for x in os.listdir(args.gpt_resume)
            if x.startswith("optimizer.") and x.endswith(".pth")
        ])
        assert opt_state_world_size == dist.get_world_size(), (
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
        )
        logger.info(f"Resuming optimizer states from: {args.gpt_resume}")
        loaded_optim_state = torch.load(os.path.join(
            args.gpt_resume,
            f"optimizer.{dist.get_rank():05d}-of-"
            f"{dist.get_world_size():05d}.pth",
        ), map_location="cpu")
        optimizer.load_state_dict(loaded_optim_state)
        # Checkpoint hyperparameters can restore fused=True even if optimizer
        # was constructed with fused=False. Force-disable it for safe resume.
        if _disable_fused_adam(optimizer):
            logger.info("Disabled fused AdamW after optimizer resume")

    # ======================================================
    #     Initialize Dataloaders
    # ======================================================
    train_dataset = build_ca_code(args, split='train')
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=global_rank,
        shuffle=True,
        seed=args.global_seed
    )
    collate_fn = custom_collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    logger.info(f"Train Dataset contains {len(train_dataset):,} CA images ({args.code_path})")

    # Validation data
    val_dirs = args.val_dirs.split(',')
    val_loaders = {}
    for val_dir in val_dirs:
        val_dataset = build_ca_code(args, split=val_dir)
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=global_rank,
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
            drop_last=True,
            collate_fn=collate_fn
        )
        val_loaders[val_dir] = val_loader
        logger.info(f"Validation dataset {val_dir} contains {len(val_dataset):,} CA images")

    # ======================================================
    #   Start training
    # ======================================================
    if args.gpt_resume:
        with open(os.path.join(args.gpt_resume, "resume_step.txt")) as f:
            train_steps = int(f.read().strip())
        start_epoch = int(train_steps / int(len(train_dataset) / args.global_batch_size))
        train_steps = int(start_epoch * int(len(train_dataset) / args.global_batch_size))
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0

    model.train()

    ptdtype = {
        'fp32': torch.float32, 'tf32': torch.float32,
        'bf16': torch.bfloat16, 'fp16': torch.float16,
    }[args.mixed_precision]

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    iters_train = len(train_loader)
    max_it = args.epochs * iters_train
    wp_it = args.warmup_epochs * iters_train

    logger.info(f"Training for {args.epochs} epochs...")
    logger.info(f"wp_it={wp_it}, iters_train={iters_train}, max_it={max_it}")
    logger.info(f"lr={args.lr}, schedule={args.lr_schedule}")

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for it, (z_with_eos, y, n_channels, np_mask, lens) in enumerate(train_loader):
            g_it = epoch * iters_train + it

            if args.lr_schedule is not None:
                min_tlr, max_tlr = lr_wd_annealing(args.lr_schedule, optimizer, peak_lr=args.lr, cur_it=g_it, wp_it=wp_it, max_it=max_it, wp0=args.wp0, wpe=args.wpe)
            else:
                min_tlr, max_tlr = args.lr, args.lr

            z_with_eos = z_with_eos.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if np_mask is not None:
                np_mask = np_mask.to(device, non_blocking=True)
                lens = lens.to(device, non_blocking=True)

            z_indices = z_with_eos[:, :-1]
            targets = z_with_eos

            optimizer.zero_grad()

            # Autocast context for mixed precision
            amp_ctx = {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.mixed_precision]

            if args.gpt_type == 'ca':
                with amp_ctx:
                    _, loss = model(
                        idx=z_indices,
                        cond_idx=torch.zeros((z_indices.shape[0], 1), device=device, dtype=torch.long),
                        targets=targets
                    )
            elif args.gpt_type == 'ca_binary_prefix':
                with amp_ctx:
                    _, loss = model(
                        idx=z_indices,
                        cond_idx=y,
                        targets=targets
                    )
            elif args.gpt_type == 'ca_esm_embed_mean_pool':
                y = y.unsqueeze(1)
                with amp_ctx:
                    _, loss = model(
                        idx=z_indices,
                        cond_idx=y,
                        targets=targets
                    )
            elif args.gpt_type == 'ca_esm_embed_full':
                with amp_ctx:
                    _, loss = model(
                        idx=z_indices,
                        cond_idx=y,
                        targets=targets,
                        non_mask=np_mask,
                        lens=lens
                    )
            else:
                raise ValueError(f"Unsupported model type: {args.gpt_type}")

            loss.backward()
            if args.max_grad_norm != 0.0:
                model.clip_grad_norm_(args.max_grad_norm)
            optimizer.step()

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                if global_rank == 0 and args.use_wandb:
                    log_dict = {
                        "train/loss": avg_loss,
                        "train/log_loss": np.log(avg_loss),
                        "train/steps_per_sec": steps_per_sec,
                        "train/epoch": epoch,
                        "train/step": train_steps,
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                    }
                    wandb.log(log_dict, step=train_steps)

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
                    val_loss, v_steps = eval_ep(
                        model=model, val_loader=val_loaders[val_dir],
                        val_loss=val_losses[val_dir], val_steps=val_steps_dict[val_dir],
                        device=device, args=args, ptdtype=ptdtype
                    )

                    if v_steps > 0:
                        val_loss_tensor = torch.tensor(val_loss / v_steps, device=device)
                        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                        avg_val_loss = val_loss_tensor.item() / dist.get_world_size()
                        avg_val_losses[val_dir] = avg_val_loss
                        logger.info(f"(step={train_steps:07d}) Val Loss {val_dir}: {avg_val_loss:.4f}")

                if global_rank == 0 and args.use_wandb:
                    val_log_dict = {"val/step": train_steps}
                    for val_dir in val_dirs:
                        if val_dir in avg_val_losses:
                            val_log_dict[f"val/{val_dir}/loss"] = avg_val_losses[val_dir]
                            val_log_dict[f"val/{val_dir}/log_loss"] = np.log(avg_val_losses[val_dir])
                    wandb.log(val_log_dict, step=train_steps)

                model.train()
                torch.cuda.empty_cache()

            # Save FSDP checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}"
                os.makedirs(checkpoint_path, exist_ok=True)

                # Save consolidated model state dict (rank 0 only)
                with FSDP.state_dict_type(
                    model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    consolidated_model_state_dict = model.state_dict()
                    if global_rank == 0:
                        torch.save(
                            consolidated_model_state_dict,
                            os.path.join(checkpoint_path, "consolidated.pth")
                        )
                dist.barrier()
                del consolidated_model_state_dict
                logger.info(f"Saved consolidated model to {checkpoint_path}")

                # Save per-rank optimizer shards
                opt_state_fn = (
                    f"optimizer.{dist.get_rank():05d}-of-"
                    f"{dist.get_world_size():05d}.pth"
                )
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, opt_state_fn))
                dist.barrier()
                logger.info(f"Saved optimizer to {checkpoint_path}")

                # Save training step
                if global_rank == 0:
                    with open(os.path.join(checkpoint_path, "resume_step.txt"), "w") as f:
                        print(train_steps, file=f)
                dist.barrier()
                logger.info(f"Saved training step to {checkpoint_path}")

                # Also save to local results dir if requested
                if not args.no_local_save:
                    local_checkpoint_path = f"{experiment_dir}/checkpoints/{train_steps:07d}"
                    os.makedirs(local_checkpoint_path, exist_ok=True)
                    with FSDP.state_dict_type(
                        model,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                    ):
                        consolidated_model_state_dict = model.state_dict()
                        if global_rank == 0:
                            torch.save(
                                consolidated_model_state_dict,
                                os.path.join(local_checkpoint_path, "consolidated.pth")
                            )
                    dist.barrier()
                    del consolidated_model_state_dict

                    torch.save(optimizer.state_dict(), os.path.join(local_checkpoint_path, opt_state_fn))
                    if global_rank == 0:
                        with open(os.path.join(local_checkpoint_path, "resume_step.txt"), "w") as f:
                            print(train_steps, file=f)
                    dist.barrier()
                    logger.info(f"Saved local checkpoint to {local_checkpoint_path}")

    model.eval()
    logger.info("Done!")

    if global_rank == 0 and args.use_wandb:
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--cloud-save-path", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--experiment-name", type=str, default=None, help='experiment name to prepend to checkpoint directory')
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-resume", type=str, default=None, help="checkpoint directory for resume training (contains consolidated.pth + optimizer shards)")
    parser.add_argument("--gpt-type", type=str, choices=['ca', 'ca_binary_prefix', 'ca_esm_embed_mean_pool', 'ca_esm_embed_full'], default="ca", help="type of conditioning")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--pretrained-gpt-ckpt", type=str, default=None, help="pretrained GPT model checkpoint path (loaded on rank 0 before FSDP wrapping)")
    parser.add_argument("--ema", action='store_true', help="NOT SUPPORTED with FSDP - will raise error")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='ca_code')
    parser.add_argument("--val-dirs", type=str, default='val', help="comma-separated list of directories to use for validation")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--n-channels", type=int, default=5, help="number of channels in CA images")
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
    parser.add_argument("--mixed-precision", type=str, choices=["fp32", "tf32", "fp16", "bf16"], default='bf16')
    parser.add_argument("--data-parallel", type=str, choices=["sdp", "fsdp", "hsdp"], default="fsdp")
    parser.add_argument("--grad-precision", type=str, choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--use-wandb", action='store_true', help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="llamagen-ca", help="W&B project name")
    parser.add_argument("--wandb-run-id", type=str, default=None, help="W&B run ID for resuming")
    args = parser.parse_args()
    main(args)
