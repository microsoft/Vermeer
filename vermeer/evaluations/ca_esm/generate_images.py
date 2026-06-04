#!/usr/bin/env python3
"""
Stage 1: Image Generation Script

Load a checkpoint, generate images from validation codes, save true + generated images to disk.
Output is consumed by evaluate_subcell.py and evaluate_fid.py.

Usage:
    python evaluations/ca_esm/generate_images.py \
        --model_checkpoint /path/to/checkpoint.pt \
        --val_dir /path/to/val_data \
        --output_dir /path/to/output
"""

import numpy as np
from pathlib import Path
import argparse
import json
import torch
import sys
import os
import time
from datetime import datetime
from tqdm import tqdm
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt_ca import GPT_models
from autoregressive.models.generate_ca import (
    generate,
    generate_with_prefix,
    decode_tokens_to_images,
    decode_tokens_to_images_batched,
    load_validation_codes,
)


def load_models(args, device):
    """Load VQ tokenizer and GPT-CA model."""
    print("Loading VQ tokenizer...")
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    )
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"VQ tokenizer loaded from {args.vq_ckpt}")

    print("Loading GPT-CA model...")
    precision = torch.bfloat16
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        n_max_channels=args.n_max_channels,
        block_size_per_channel=args.block_size_per_channel,
        model_type=args.model_type,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.model_checkpoint, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        model_weight = checkpoint

    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"GPT-CA model loaded from {args.model_checkpoint}")

    return vq_model, gpt_model


def pad_and_batch_esm_embeddings(batch_labels):
    """Pad and batch ESM embeddings for variable-length sequences."""
    labels_list = batch_labels

    if isinstance(labels_list[0], np.ndarray):
        labels_list = [torch.from_numpy(label) for label in labels_list]

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

        return labels, padding_mask, lens
    else:
        labels = torch.stack(labels_list)
        return labels, None, None


def save_channel_images(images, filenames, output_dir, prefix):
    """
    Save multi-channel images as individual PNGs.

    Args:
        images: (B, 4, H, W) numpy array with channels [nucleus, microtubule, ER, protein]
        filenames: List of filenames (without .npy)
        output_dir: Output directory
        prefix: 'true' or 'gen'

    Returns:
        Number of images saved
    """
    os.makedirs(output_dir, exist_ok=True)
    channel_names = ["nucleus", "microtubule", "er", "protein"]
    count = 0

    for i, filename in enumerate(filenames):
        base_name = filename.replace(".npy", "")
        for ch_idx, ch_name in enumerate(channel_names):
            ch_data = images[i, ch_idx]
            ch_data = (
                (ch_data - ch_data.min()) / (ch_data.max() - ch_data.min() + 1e-8) * 255
            ).astype(np.uint8)
            path = os.path.join(output_dir, f"{prefix}_{base_name}_{ch_name}.png")
            Image.fromarray(ch_data).save(path)
            count += 1

    return count


def check_existing_images(images_dir_true, images_dir_gen, filenames, generation_mode="protein_only"):
    """Check if all expected images already exist."""
    all_channels = ["nucleus", "microtubule", "er", "protein"]
    # In protein_only mode, true images have all 4 channels but gen only has all 4
    # (we decode all channels for both true and gen regardless of mode)
    true_channels = all_channels
    gen_channels = all_channels
    missing = []

    for filename in filenames:
        base_name = filename.replace(".npy", "")
        true_ok = all(
            os.path.exists(os.path.join(images_dir_true, f"true_{base_name}_{ch}.png"))
            for ch in true_channels
        )
        gen_ok = all(
            os.path.exists(os.path.join(images_dir_gen, f"gen_{base_name}_{ch}.png"))
            for ch in gen_channels
        )
        if not (true_ok and gen_ok):
            missing.append(filename)

    return len(missing) == 0, missing


def parse_args():
    parser = argparse.ArgumentParser(description="Generate true and predicted images from a CA checkpoint")

    # Required
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to GPT-CA model checkpoint")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for images and metadata")

    # Generation mode
    parser.add_argument("--generation_mode", type=str, default="protein_only",
                        choices=["protein_only", "all_channels"],
                        help="protein_only: generate protein channel conditioned on landmark stains. "
                             "all_channels: generate all channels from ESM embedding only (default: protein_only)")

    # Data config
    parser.add_argument("--val_split", type=str, default="val1", help="Validation split (default: val1)")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples (None = all)")

    # Model config
    parser.add_argument("--vq_ckpt", type=str,
                        default="/n/holylabs/chenf2011_lab/Everyone/Sandeep/code/CA_LlamaGen/pretrained_models/vq_ds16_c2i.pt")  # change to your path
    parser.add_argument("--vq_model", type=str, default="VQ-16")
    parser.add_argument("--gpt_model", type=str, default="GPT-B")
    parser.add_argument("--model_type", type=str, default="ca_esm_embed_mean_pool")
    parser.add_argument("--codebook_size", type=int, default=16384)
    parser.add_argument("--codebook_embed_dim", type=int, default=8)
    parser.add_argument("--n_channels", type=int, default=4)
    parser.add_argument("--n_max_channels", type=int, default=5)
    parser.add_argument("--block_size_per_channel", type=int, default=256)
    parser.add_argument("--n_prefix_channels", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--downsample_size", type=int, default=16)

    # Generation config
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for AR generation")
    parser.add_argument("--decode_batch_size", type=int, default=64, help="Batch size for VQ decoding of true images")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=2000)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Stage 1: Image Generation")
    print("=" * 60)
    print(f"Generation mode: {args.generation_mode}")
    print(f"Model checkpoint: {args.model_checkpoint}")
    print(f"Validation directory: {args.val_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")

    # Setup output directories
    output_dir = Path(args.output_dir)
    images_dir_true = output_dir / "images_true"
    images_dir_gen = output_dir / "images_gen"
    metadata_path = output_dir / "generation_metadata.json"

    images_dir_true.mkdir(parents=True, exist_ok=True)
    images_dir_gen.mkdir(parents=True, exist_ok=True)

    # Load validation codes
    code_dir = os.path.join(args.val_dir, f"ca{args.image_size}_codes", args.val_split)
    label_dir = os.path.join(args.val_dir, f"ca{args.image_size}_labels", args.val_split)

    print(f"\nLoading validation data from {code_dir}...")
    codes, labels, filenames = load_validation_codes(code_dir, label_dir, aug_idx=0)

    if args.num_samples is not None:
        codes = codes[: args.num_samples]
        labels = labels[: args.num_samples]
        filenames = filenames[: args.num_samples]

    print(f"Loaded {len(filenames)} samples")

    # Check if all images already exist
    all_exist, missing = check_existing_images(str(images_dir_true), str(images_dir_gen), filenames, args.generation_mode)

    if all_exist and metadata_path.exists():
        print("\nAll images already exist and metadata found. Skipping generation.")
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.1f}s")
        return

    if len(missing) < len(filenames):
        print(f"\n{len(filenames) - len(missing)}/{len(filenames)} samples already exist. Generating remaining {len(missing)}.")

    # Load models
    vq_model, gpt_model = load_models(args, device)

    if not args.no_compile:
        print("Compiling GPT model with torch.compile...")
        gpt_model = torch.compile(gpt_model)

    # Generation loop
    tokens_per_channel = args.block_size_per_channel + 2
    latent_size = args.image_size // args.downsample_size
    all_filenames = []

    # Decode true images in large batches (faster since no AR generation needed)
    print(f"\nDecoding true images (decode_batch_size={args.decode_batch_size})...")
    for batch_start in tqdm(range(0, len(codes), args.decode_batch_size), desc="Decoding true images"):
        batch_end = min(batch_start + args.decode_batch_size, len(codes))
        batch_codes = codes[batch_start:batch_end]
        batch_filenames = filenames[batch_start:batch_end]

        # Check if this batch already exists
        batch_all_exist = all(
            all(
                os.path.exists(os.path.join(images_dir_true, f"true_{fn.replace('.npy', '')}_{ch}.png"))
                for ch in ["nucleus", "microtubule", "er", "protein"]
            )
            for fn in batch_filenames
        )
        if batch_all_exist:
            continue

        full_codes = []
        for code in batch_codes:
            if len(code.shape) > 1:
                code = code[0]
            full_codes.append(code)

        true_tokens = torch.from_numpy(np.stack(full_codes)).long().to(device)
        true_images = decode_tokens_to_images_batched(
            true_tokens, vq_model, args.n_channels, tokens_per_channel, latent_size, args.codebook_embed_dim
        )
        true_np = true_images.mean(dim=2).cpu().numpy()
        save_channel_images(true_np, batch_filenames, str(images_dir_true), prefix="true")

    # Generate images via AR model
    mode_desc = "protein channel (landmark-conditioned)" if args.generation_mode == "protein_only" else "all channels (unconditional)"
    print(f"\nGenerating {mode_desc} (batch_size={args.batch_size})...")
    for batch_start in tqdm(range(0, len(codes), args.batch_size), desc="Generating samples"):
        batch_end = min(batch_start + args.batch_size, len(codes))
        batch_codes = codes[batch_start:batch_end]
        batch_labels = labels[batch_start:batch_end]
        batch_filenames = filenames[batch_start:batch_end]

        # Check if this batch already exists
        batch_all_exist = all(
            all(
                os.path.exists(os.path.join(images_dir_gen, f"gen_{fn.replace('.npy', '')}_{ch}.png"))
                for ch in ["nucleus", "microtubule", "er", "protein"]
            )
            for fn in batch_filenames
        )
        if batch_all_exist:
            all_filenames.extend(batch_filenames)
            continue

        # Prepare codes
        flat_codes = []
        for code in batch_codes:
            if len(code.shape) > 1:
                code = code[0]
            flat_codes.append(code)

        # Prepare ESM embeddings
        if args.model_type == "ca_esm_embed_mean_pool":
            cond = torch.from_numpy(np.stack(batch_labels)).float().unsqueeze(1).to(device)
            padding_mask = None
            lens = None
        elif args.model_type == "ca_esm_embed_full":
            cond, padding_mask, lens = pad_and_batch_esm_embeddings(batch_labels)
            cond = cond.float().to(device)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)
            if lens is not None:
                lens = lens.to(device)
        else:
            cond = torch.from_numpy(np.stack(batch_labels)).float().unsqueeze(1).to(device)
            padding_mask = None
            lens = None

        cond = cond.to(dtype=next(gpt_model.parameters()).dtype)

        if args.generation_mode == "protein_only":
            # Generate protein channel conditioned on landmark stains
            prefix_len = args.n_prefix_channels * tokens_per_channel
            prefix_tokens = [code[:prefix_len] for code in flat_codes]
            prefix_tokens = torch.from_numpy(np.stack(prefix_tokens)).long().to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                generated_tokens = generate_with_prefix(
                    gpt_model,
                    cond,
                    prefix_tokens,
                    n_total_channels=args.n_channels,
                    tokens_per_channel=tokens_per_channel,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    padding_mask=padding_mask,
                    lens=lens,
                    sample_logits=True,
                )

            # Remove EOS token
            generated_tokens = generated_tokens[:, :-1]

        else:
            # all_channels mode: generate all channels from ESM embedding only
            max_new_tokens = args.n_channels * tokens_per_channel + 1  # +1 for EOS

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                generated_tokens = generate(
                    gpt_model,
                    cond,
                    max_new_tokens=max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    sample_logits=True,
                )

            # Remove EOS token
            generated_tokens = generated_tokens[:, :-1]

        # Decode generated images
        generated_images = decode_tokens_to_images(
            generated_tokens, vq_model, args.n_channels, tokens_per_channel, latent_size, args.codebook_embed_dim
        )
        gen_np = generated_images.mean(dim=2).cpu().numpy()
        save_channel_images(gen_np, batch_filenames, str(images_dir_gen), prefix="gen")

        all_filenames.extend(batch_filenames)

    # If we skipped some batches that already existed, make sure filenames is complete
    all_filenames = filenames

    # Save metadata
    elapsed = time.time() - start_time
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "filenames": all_filenames,
        "num_samples": len(all_filenames),
        "generation_mode": args.generation_mode,
        "configuration": {
            "model_checkpoint": args.model_checkpoint,
            "val_dir": args.val_dir,
            "val_split": args.val_split,
            "gpt_model": args.gpt_model,
            "model_type": args.model_type,
            "generation_mode": args.generation_mode,
            "n_channels": args.n_channels,
            "n_prefix_channels": args.n_prefix_channels,
            "image_size": args.image_size,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "timing": {"total_seconds": elapsed},
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGeneration complete: {len(all_filenames)} samples")
    print(f"Images saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
