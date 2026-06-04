#!/usr/bin/env python3
"""
Evaluate teacher-forcing validation cross-entropy on protein-channel patch tokens.

This script mirrors the training validation data path used in `train_ca.py`:
- dataset from `build_ca_code(...)`
- collate behavior from `make_collate_fn(...)`
- teacher-forcing forward pass with `idx=z_with_eos[:, :-1]`, `targets=z_with_eos`

Unlike training's full-sequence loss, this computes CE only on protein patch tokens
(excluding SOC/EOC), then reports per-split and aggregate metrics.

Usage:
    python evaluations/ca_esm/evaluate_protein_val_loss.py \
        --model_checkpoint /path/to/checkpoint.pt \
        --code_path /path/to/code_root \
        --val_splits val1,val2 \
        --output_json /path/to/protein_val_loss.json
"""

import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from autoregressive.models.gpt_ca import GPT_models
from dataset.ca_image import build_ca_code


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate teacher-forcing validation CE on protein patch tokens only"
    )

    # Required
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to GPT checkpoint to evaluate",
    )
    parser.add_argument(
        "--code_path",
        type=str,
        required=True,
        help="Root directory containing ca{image_size}_codes/ and ca{image_size}_labels/",
    )

    # Dataset / split config
    parser.add_argument(
        "--val_splits",
        type=str,
        default="val1",
        help="Comma-separated validation splits (e.g. val1,val2)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size used for extracted codes",
    )
    parser.add_argument(
        "--input_res",
        type=int,
        default=None,
        help="Image size used for extracted codes",
    )
    parser.add_argument(
        "--downsample_size",
        type=int,
        default=16,
        help="VQ downsample size used for extracted codes",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Dataloader workers",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Validation batch size",
    )
    parser.add_argument(
        "--keep_last",
        action="store_true",
        help="Keep final incomplete batch (default behavior matches training parity and drops it)",
    )

    # Model config
    parser.add_argument("--gpt_model", type=str, default="GPT-B")
    parser.add_argument(
        "--gpt_type",
        type=str,
        default="ca_esm_embed_mean_pool",
        choices=["ca", "ca_binary_prefix", "ca_esm_embed_mean_pool", "ca_esm_embed_full"],
    )
    parser.add_argument("--vocab_size", type=int, default=16384)
    parser.add_argument(
        "--n_channels",
        type=int,
        default=4,
        help="Number of channels in dataset before optional channel_config",
    )
    parser.add_argument(
        "--n_max_channels",
        type=int,
        default=5,
        help="Max channels model supports (embedding table width)",
    )
    parser.add_argument(
        "--block_size_per_channel",
        type=int,
        default=None,
        help="Patch tokens per channel; defaults to (image_size/downsample_size)^2",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="Kept for GPT constructor parity",
    )
    parser.add_argument(
        "--cls_token_num",
        type=int,
        default=1,
        help="Condition token count in GPT constructor",
    )
    parser.add_argument(
        "--channel_config",
        type=str,
        default=None,
        help="Optional comma-separated channel indices to select/reorder, e.g. '0,1,3'",
    )
    parser.add_argument(
        "--protein_channel_idx",
        type=int,
        default=3,
        help="Protein channel index in original channel numbering",
    )

    # Runtime / output
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["none", "fp16", "bf16"],
        help="Autocast precision for forward pass on CUDA",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON path (default: ./protein_val_loss_results.json)",
    )
    return parser.parse_args()


def get_ptdtype(mixed_precision):
    if mixed_precision == "none":
        return None
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed_precision: {mixed_precision}")


def parse_channel_config(channel_config, n_channels):
    if channel_config is None:
        return None
    parsed = [int(x) for x in channel_config.split(",") if x.strip() != ""]
    if len(parsed) == 0:
        raise ValueError("channel_config is empty after parsing")
    invalid = [c for c in parsed if c < 0 or c >= n_channels]
    if invalid:
        raise ValueError(
            f"channel_config has invalid indices {invalid}; expected range [0, {n_channels - 1}]"
        )
    return parsed


def resolve_selected_channels(n_channels, channel_config):
    if channel_config is None:
        return list(range(n_channels))
    return list(channel_config)


def apply_channel_config(features_batch, n_channels, channel_config, tokens_per_channel=258):
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


def make_collate_fn(n_channels=4, tokens_per_channel=258, channel_config=None):
    def collate_fn(batch):
        features = torch.stack([item[0] for item in batch])
        labels_list = [item[1] for item in batch]
        n_channels_batch = batch[0][2]
        if n_channels is not None and n_channels_batch != n_channels:
            raise ValueError(
                f"Batch n_channels ({n_channels_batch}) does not match expected n_channels ({n_channels}). "
                "Check dataset configuration and evaluation arguments."
            )

        if channel_config is not None:
            features, n_channels_batch = apply_channel_config(
                features, n_channels_batch, channel_config, tokens_per_channel
            )

        if len(labels_list[0].shape) > 1:
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
            return features, labels, n_channels_batch, padding_mask, lens

        labels = torch.stack(labels_list)
        return features, labels, n_channels_batch, None, None

    return collate_fn


def load_gpt_model(args, device, ptdtype):
    block_size_per_channel = args.block_size_per_channel
    if block_size_per_channel is None:
        latent_size = args.image_size // args.downsample_size
        block_size_per_channel = latent_size**2

    if args.gpt_model not in GPT_models:
        raise ValueError(
            f"Unknown gpt_model={args.gpt_model}. Available: {list(GPT_models.keys())}"
        )

    model_dtype = torch.float32 if ptdtype is None else ptdtype
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size_per_channel=block_size_per_channel,
        n_max_channels=args.n_max_channels,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=model_dtype)

    checkpoint = torch.load(args.model_checkpoint, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        model_weight = checkpoint

    model.load_state_dict(model_weight, strict=False)
    model.eval()
    return model, block_size_per_channel


def build_val_loader(args, split, tokens_per_channel, channel_config):
    args_for_dataset = argparse.Namespace(**vars(args))
    args_for_dataset.code_path = args.code_path
    args_for_dataset.input_res = args.input_res
    dataset = build_ca_code(args_for_dataset, split=split)
    collate_fn = make_collate_fn(
        n_channels=args.n_channels,
        tokens_per_channel=tokens_per_channel,
        channel_config=channel_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=not args.keep_last,
        collate_fn=collate_fn,
    )
    return dataset, loader


def forward_logits_for_batch(model, args, z_indices, y, np_mask, lens, autocast_context):
    if args.gpt_type == "ca":
        cond = torch.zeros((z_indices.shape[0], 1), device=z_indices.device, dtype=torch.long)
        with autocast_context:
            logits, _ = model(idx=z_indices, cond_idx=cond, targets=None)
        return logits

    if args.gpt_type == "ca_binary_prefix":
        with autocast_context:
            logits, _ = model(idx=z_indices, cond_idx=y, targets=None)
        return logits

    if args.gpt_type == "ca_esm_embed_mean_pool":
        with autocast_context:
            logits, _ = model(idx=z_indices, cond_idx=y.unsqueeze(1), targets=None)
        return logits

    if args.gpt_type == "ca_esm_embed_full":
        with autocast_context:
            logits, _ = model(
                idx=z_indices,
                cond_idx=y,
                targets=None,
                non_mask=np_mask,
                lens=lens,
            )
        return logits

    raise ValueError(f"Unsupported gpt_type: {args.gpt_type}")


def compute_split_protein_loss(
    model,
    loader,
    device,
    args,
    protein_pos_in_selected,
    tokens_per_channel,
    block_size_per_channel,
    ptdtype,
):
    protein_patch_start = protein_pos_in_selected * tokens_per_channel + 1
    protein_patch_end = protein_patch_start + block_size_per_channel

    total_loss_sum = 0.0
    total_token_count = 0
    total_samples = 0
    total_steps = 0

    if device.type == "cuda" and ptdtype is not None:
        autocast_context = torch.amp.autocast("cuda", dtype=ptdtype)
    else:
        autocast_context = nullcontext()

    with torch.no_grad():
        for z_with_eos, y, _n_channels_batch, np_mask, lens in tqdm(
            loader, desc="Evaluating", leave=False
        ):
            z_with_eos = z_with_eos.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if np_mask is not None:
                np_mask = np_mask.to(device, non_blocking=True)
            if lens is not None:
                lens = lens.to(device, non_blocking=True)

            z_indices = z_with_eos[:, :-1]
            targets = z_with_eos
            logits = forward_logits_for_batch(
                model=model,
                args=args,
                z_indices=z_indices,
                y=y,
                np_mask=np_mask,
                lens=lens,
                autocast_context=autocast_context,
            )

            if logits.shape[1] != targets.shape[1]:
                raise ValueError(
                    f"Logit/target sequence mismatch: logits={logits.shape}, targets={targets.shape}. "
                    "This script expects teacher-forcing logits aligned to target length."
                )

            if protein_patch_end > targets.shape[1]:
                raise ValueError(
                    f"Protein patch range [{protein_patch_start}, {protein_patch_end}) exceeds "
                    f"target length {targets.shape[1]}."
                )

            per_token_ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            ).reshape(targets.shape[0], targets.shape[1])

            token_mask = torch.zeros(
                (targets.shape[0], targets.shape[1]), dtype=torch.bool, device=targets.device
            )
            token_mask[:, protein_patch_start:protein_patch_end] = True

            masked_loss = per_token_ce[token_mask]
            token_count = int(token_mask.sum().item())
            if token_count == 0:
                raise ValueError("Protein token mask selected zero tokens for a batch.")

            total_loss_sum += float(masked_loss.sum().item())
            total_token_count += token_count
            total_samples += targets.shape[0]
            total_steps += 1

    if total_token_count == 0:
        raise ValueError("No protein tokens were selected across validation data.")

    mean_loss = total_loss_sum / total_token_count
    return {
        "protein_val_loss": float(mean_loss),
        "sum_ce_protein_tokens": float(total_loss_sum),
        "count_protein_tokens": int(total_token_count),
        "num_samples_evaluated": int(total_samples),
        "num_steps": int(total_steps),
        "protein_patch_start_idx": int(protein_patch_start),
        "protein_patch_end_idx_exclusive": int(protein_patch_end),
    }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ptdtype = get_ptdtype(args.mixed_precision)

    channel_config = parse_channel_config(args.channel_config, args.n_channels)
    selected_channels = resolve_selected_channels(args.n_channels, channel_config)
    if args.protein_channel_idx < 0 or args.protein_channel_idx >= args.n_channels:
        raise ValueError(
            f"protein_channel_idx={args.protein_channel_idx} out of range for n_channels={args.n_channels}"
        )
    if args.protein_channel_idx not in selected_channels:
        raise ValueError(
            f"protein_channel_idx={args.protein_channel_idx} not present after channel_config={selected_channels}"
        )
    protein_pos_in_selected = selected_channels.index(args.protein_channel_idx)

    model, block_size_per_channel = load_gpt_model(args, device, ptdtype)
    if not args.no_compile:
        model = torch.compile(model)

    val_splits = [s.strip() for s in args.val_splits.split(",") if s.strip()]
    if len(val_splits) == 0:
        raise ValueError("val_splits is empty after parsing.")

    tokens_per_channel = block_size_per_channel + 2

    print("=" * 70)
    print("Protein Channel Validation Loss (Teacher-Forcing CE)")
    print("=" * 70)
    print(f"Model checkpoint: {args.model_checkpoint}")
    print(f"Code path: {args.code_path}")
    print(f"Validation splits: {val_splits}")
    print(f"Device: {device}")
    print(f"gpt_type: {args.gpt_type}")
    print(f"selected_channels: {selected_channels}")
    print(
        f"Protein channel original idx={args.protein_channel_idx}, "
        f"position in selected sequence={protein_pos_in_selected}"
    )
    print(
        f"block_size_per_channel={block_size_per_channel}, tokens_per_channel={tokens_per_channel}"
    )
    print(f"drop_last={not args.keep_last}")
    print("=" * 70)

    split_results = {}
    aggregate_sum = 0.0
    aggregate_count = 0
    aggregate_samples = 0
    aggregate_steps = 0

    for split in val_splits:
        dataset, loader = build_val_loader(
            args=args,
            split=split,
            tokens_per_channel=tokens_per_channel,
            channel_config=channel_config,
        )
        print(f"\nEvaluating split='{split}' with {len(dataset)} samples...")
        split_metrics = compute_split_protein_loss(
            model=model,
            loader=loader,
            device=device,
            args=args,
            protein_pos_in_selected=protein_pos_in_selected,
            tokens_per_channel=tokens_per_channel,
            block_size_per_channel=block_size_per_channel,
            ptdtype=ptdtype,
        )
        split_metrics["dataset_len"] = int(len(dataset))
        split_results[split] = split_metrics

        aggregate_sum += split_metrics["sum_ce_protein_tokens"]
        aggregate_count += split_metrics["count_protein_tokens"]
        aggregate_samples += split_metrics["num_samples_evaluated"]
        aggregate_steps += split_metrics["num_steps"]

        print(
            f"  protein_val_loss={split_metrics['protein_val_loss']:.6f} "
            f"(tokens={split_metrics['count_protein_tokens']}, samples={split_metrics['num_samples_evaluated']})"
        )

    if aggregate_count == 0:
        raise ValueError("Aggregate protein token count is zero.")

    aggregate_loss = aggregate_sum / aggregate_count
    print("\n" + "=" * 70)
    print("Aggregate protein validation loss")
    print("=" * 70)
    print(f"protein_val_loss={aggregate_loss:.6f}")
    print(f"total_protein_tokens={aggregate_count}")
    print(f"total_samples={aggregate_samples}")
    print(f"total_steps={aggregate_steps}")
    print("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "model_checkpoint": args.model_checkpoint,
            "code_path": args.code_path,
            "val_splits": val_splits,
            "gpt_model": args.gpt_model,
            "gpt_type": args.gpt_type,
            "vocab_size": args.vocab_size,
            "n_channels": args.n_channels,
            "n_max_channels": args.n_max_channels,
            "channel_config": channel_config,
            "selected_channels": selected_channels,
            "protein_channel_idx": args.protein_channel_idx,
            "protein_channel_position_in_selected": protein_pos_in_selected,
            "image_size": args.image_size,
            "downsample_size": args.downsample_size,
            "block_size_per_channel": block_size_per_channel,
            "tokens_per_channel": tokens_per_channel,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "drop_last": not args.keep_last,
            "mixed_precision": args.mixed_precision,
            "seed": args.seed,
            "compiled": not args.no_compile,
        },
        "metrics": {
            "aggregate": {
                "protein_val_loss": float(aggregate_loss),
                "sum_ce_protein_tokens": float(aggregate_sum),
                "count_protein_tokens": int(aggregate_count),
                "num_samples_evaluated": int(aggregate_samples),
                "num_steps": int(aggregate_steps),
            },
            "per_split": split_results,
        },
    }

    output_json = args.output_json or os.path.abspath("./protein_val_loss_results.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to: {output_json}")


if __name__ == "__main__":
    main()
