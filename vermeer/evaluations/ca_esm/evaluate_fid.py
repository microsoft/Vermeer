#!/usr/bin/env python3
"""
Stage 3: FID Evaluation

Compute FID scores on pre-generated images (per-channel).
Expects images from generate_images.py (Stage 1).

Usage:
    python evaluations/ca_esm/evaluate_fid.py \
        --images_dir /path/to/output_from_stage1 \
        --output_json /path/to/fid_results.json
"""

import argparse
import os
import sys
import glob
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


def load_png_image(image_path):
    """Load a PNG image and convert to (3, H, W) uint8 tensor for FID."""
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)  # (H, W)
    img_rgb = np.stack([img_array, img_array, img_array], axis=0)  # (3, H, W)
    return torch.from_numpy(img_rgb).byte()


def _extract_basename(img_path, channel):
    """Extract sample basename from an image path, stripping prefix and channel suffix."""
    basename = Path(img_path).stem
    if basename.startswith("true_"):
        basename = basename[5:]
    elif basename.startswith("gen_"):
        basename = basename[4:]
    suffix = f"_{channel}"
    if basename.endswith(suffix):
        basename = basename[:-len(suffix)]
    return basename


def load_images_from_directory(images_dir, channel="protein", verbose=True):
    """Load all images of a specific channel from a directory."""
    images_dir = Path(images_dir)
    pattern = f"*_{channel}.png"
    image_files = sorted(glob.glob(str(images_dir / pattern)))

    if len(image_files) == 0:
        raise FileNotFoundError(f"No images matching '{pattern}' in {images_dir}")

    if verbose:
        print(f"Found {len(image_files)} {channel} images in {images_dir}")

    images_list = []
    filenames_list = []

    for img_path in tqdm(image_files, desc=f"Loading {channel} images", disable=not verbose):
        images_list.append(load_png_image(img_path))
        filenames_list.append(_extract_basename(img_path, channel))

    return torch.stack(images_list, dim=0), filenames_list


def scan_filenames(images_dir, channel="protein", verbose=True):
    """Scan directory for image paths and basenames without loading pixels."""
    images_dir = Path(images_dir)
    pattern = f"*_{channel}.png"
    image_files = sorted(glob.glob(str(images_dir / pattern)))

    if len(image_files) == 0:
        raise FileNotFoundError(f"No images matching '{pattern}' in {images_dir}")

    if verbose:
        print(f"Found {len(image_files)} {channel} images in {images_dir}")

    filenames = [_extract_basename(p, channel) for p in image_files]
    return image_files, filenames


def match_image_pairs(true_filenames, gen_filenames):
    """Match true and generated images by filename."""
    true_map = {fname: idx for idx, fname in enumerate(true_filenames)}
    gen_map = {fname: idx for idx, fname in enumerate(gen_filenames)}
    common_files = set(true_filenames) & set(gen_filenames)

    if len(common_files) == 0:
        raise ValueError("No matching filenames found between true and generated images")

    true_indices = [true_map[fname] for fname in sorted(common_files)]
    gen_indices = [gen_map[fname] for fname in sorted(common_files)]
    return true_indices, gen_indices, sorted(common_files)


def compute_fid_score(real_images, generated_images, batch_size=32, device="cuda"):
    """Compute FID score between real and generated images."""
    print(f"  Real images: {real_images.shape}, Generated images: {generated_images.shape}")

    fid_metric = FrechetInceptionDistance().to(device)

    for i in tqdm(range(0, real_images.shape[0], batch_size), desc="  Processing real"):
        fid_metric.update(real_images[i : i + batch_size].to(device), real=True)

    for i in tqdm(range(0, generated_images.shape[0], batch_size), desc="  Processing generated"):
        fid_metric.update(generated_images[i : i + batch_size].to(device), real=False)

    return fid_metric.compute().item()


def compute_fid_score_streaming(true_paths, gen_paths, batch_size=32, device="cuda"):
    """Compute FID by streaming images from disk in batches (low memory)."""
    n = len(true_paths)
    print(f"  Streaming FID over {n} matched pairs (batch_size={batch_size})")

    fid_metric = FrechetInceptionDistance().to(device)

    # Feed real images
    for i in tqdm(range(0, n, batch_size), desc="  Processing real"):
        batch = torch.stack(
            [load_png_image(p) for p in true_paths[i : i + batch_size]]
        )
        fid_metric.update(batch.to(device), real=True)
        del batch

    # Feed generated images
    for i in tqdm(range(0, n, batch_size), desc="  Processing generated"):
        batch = torch.stack(
            [load_png_image(p) for p in gen_paths[i : i + batch_size]]
        )
        fid_metric.update(batch.to(device), real=False)
        del batch

    return fid_metric.compute().item()


def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID scores on pre-generated images")

    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing images_true/ and images_gen/")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Output JSON path (default: {images_dir}/fid_results.json)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for FID computation")
    parser.add_argument("--channel", type=str, default=None,
                        choices=["protein", "nucleus", "microtubule", "er"],
                        help="Evaluate single channel (default: all)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Stage 3: FID Evaluation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    images_dir = Path(args.images_dir)
    images_dir_true = images_dir / "images_true"
    images_dir_gen = images_dir / "images_gen"

    if not images_dir_true.exists() or not images_dir_gen.exists():
        raise FileNotFoundError(
            f"images_true/ or images_gen/ not found in {images_dir}. Run generate_images.py first."
        )

    # Load metadata if available
    metadata_path = images_dir / "generation_metadata.json"
    config = {}
    generation_mode = "protein_only"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        config = metadata.get("configuration", {})
        generation_mode = metadata.get("generation_mode", "protein_only")

    print(f"Generation mode: {generation_mode}")

    # Determine channels to evaluate
    if args.channel:
        channels = [args.channel]
    elif generation_mode == "all_channels":
        # All channels were generated, evaluate all
        channels = ["protein", "nucleus", "microtubule", "er"]
    else:
        # protein_only mode: only the protein channel was generated,
        # landmark channels are copied from ground truth so FID is only meaningful for protein
        channels = ["protein"]
        print("protein_only mode: evaluating FID on protein channel only")

    all_results = {}

    for channel in channels:
        print(f"\nEvaluating channel: {channel.upper()}")

        true_images, true_filenames = load_images_from_directory(images_dir_true, channel=channel)
        gen_images, gen_filenames = load_images_from_directory(images_dir_gen, channel=channel)

        true_indices, gen_indices, common_files = match_image_pairs(true_filenames, gen_filenames)
        print(f"  Matched {len(common_files)} pairs")

        fid = compute_fid_score(
            true_images[true_indices],
            gen_images[gen_indices],
            batch_size=args.batch_size,
            device=device,
        )

        all_results[channel] = {
            "fid_score": fid,
            "num_matched_pairs": len(common_files),
            "num_true_images": len(true_filenames),
            "num_gen_images": len(gen_filenames),
        }

        print(f"  {channel} FID: {fid:.4f}")

    # Print summary
    print("\n" + "=" * 60)
    print("FID RESULTS")
    print("=" * 60)
    for channel in channels:
        print(f"  {channel:12s} FID: {all_results[channel]['fid_score']:.4f}")
    print(f"  Matched pairs: {all_results[channels[0]]['num_matched_pairs']}")
    print("=" * 60)

    # Save results as JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": config,
        "metrics": all_results,
    }

    output_path = args.output_json or str(images_dir / "fid_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
