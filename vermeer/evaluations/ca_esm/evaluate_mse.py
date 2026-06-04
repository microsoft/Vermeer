#!/usr/bin/env python3
"""
Stage 4: MSE Evaluation

Compute per-channel MSE scores on pre-generated images.
Expects images from generate_images.py (Stage 1).

Usage:
    python evaluations/ca_esm/evaluate_mse.py \
        --images_dir /path/to/output_from_stage1 \
        --output_json /path/to/mse_results.json
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


def load_png_image(image_path):
    """Load a PNG image and normalize to [0, 1] float32."""
    img = Image.open(image_path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def _extract_basename(img_path, channel):
    """Extract sample basename from an image path, stripping prefix and channel suffix."""
    basename = Path(img_path).stem
    if basename.startswith("true_"):
        basename = basename[5:]
    elif basename.startswith("gen_"):
        basename = basename[4:]
    suffix = f"_{channel}"
    if basename.endswith(suffix):
        basename = basename[: -len(suffix)]
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

    return np.stack(images_list, axis=0), filenames_list


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


def compute_mse_streaming(true_paths, gen_paths):
    """Compute per-image MSE by loading one pair at a time (low memory)."""
    n = len(true_paths)
    mse_vals = np.empty(n, dtype=np.float64)
    for i in tqdm(range(n), desc="  Computing MSE"):
        true_img = load_png_image(true_paths[i])
        gen_img = load_png_image(gen_paths[i])
        mse_vals[i] = np.mean((true_img - gen_img) ** 2)
    return mse_vals


def match_image_pairs(true_filenames, gen_filenames):
    """Match true and generated images by filename."""
    true_map = {fname: idx for idx, fname in enumerate(true_filenames)}
    gen_map = {fname: idx for idx, fname in enumerate(gen_filenames)}
    common_files = set(true_filenames) & set(gen_filenames)

    if len(common_files) == 0:
        raise ValueError("No matching filenames found between true and generated images")

    matched_files = sorted(common_files)
    true_indices = [true_map[fname] for fname in matched_files]
    gen_indices = [gen_map[fname] for fname in matched_files]
    return true_indices, gen_indices, matched_files


def compute_per_image_mse(true_images, gen_images):
    """Compute per-image MSE for already matched arrays of shape (N, H, W)."""
    if true_images.shape != gen_images.shape:
        raise ValueError(
            f"Shape mismatch between true and generated images: "
            f"{true_images.shape} vs {gen_images.shape}"
        )
    return np.mean((true_images - gen_images) ** 2, axis=(1, 2))


def parse_args():
    parser = argparse.ArgumentParser(description="Compute MSE scores on pre-generated images")
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing images_true/ and images_gen/",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON path (default: {images_dir}/mse_results.json)",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        choices=["protein", "nucleus", "microtubule", "er"],
        help="Evaluate single channel (default: auto from generation mode)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Stage 4: MSE Evaluation")
    print("=" * 60)

    images_dir = Path(args.images_dir)
    images_dir_true = images_dir / "images_true"
    images_dir_gen = images_dir / "images_gen"

    if not images_dir_true.exists() or not images_dir_gen.exists():
        raise FileNotFoundError(
            f"images_true/ or images_gen/ not found in {images_dir}. Run generate_images.py first."
        )

    # Load metadata if available.
    metadata_path = images_dir / "generation_metadata.json"
    config = {}
    generation_mode = "protein_only"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        config = metadata.get("configuration", {})
        generation_mode = metadata.get("generation_mode", "protein_only")

    print(f"Generation mode: {generation_mode}")

    # Determine channels to evaluate.
    if args.channel:
        channels = [args.channel]
    elif generation_mode == "all_channels":
        channels = ["protein", "nucleus", "microtubule", "er"]
    else:
        # In protein_only mode only protein was generated;
        # landmark channels are copied from ground truth.
        channels = ["protein"]
        print("protein_only mode: evaluating MSE on protein channel only")

    all_results = {}

    for channel in channels:
        print(f"\nEvaluating channel: {channel.upper()}")
        true_images, true_filenames = load_images_from_directory(images_dir_true, channel=channel)
        gen_images, gen_filenames = load_images_from_directory(images_dir_gen, channel=channel)

        true_indices, gen_indices, matched_files = match_image_pairs(true_filenames, gen_filenames)
        print(f"  Matched {len(matched_files)} pairs")

        matched_true = true_images[true_indices]
        matched_gen = gen_images[gen_indices]
        per_image_mse = compute_per_image_mse(matched_true, matched_gen)

        per_image_metrics = {
            fname: float(mse_val) for fname, mse_val in zip(matched_files, per_image_mse)
        }

        all_results[channel] = {
            "mean_mse": float(np.mean(per_image_mse)),
            "std_mse": float(np.std(per_image_mse)),
            "num_matched_pairs": len(matched_files),
            "num_true_images": len(true_filenames),
            "num_gen_images": len(gen_filenames),
            "per_image_mse": per_image_metrics,
        }

        print(
            f"  {channel} MSE: {all_results[channel]['mean_mse']:.6f} "
            f"+/- {all_results[channel]['std_mse']:.6f}"
        )

    print("\n" + "=" * 60)
    print("MSE RESULTS")
    print("=" * 60)
    for channel in channels:
        print(
            f"  {channel:12s} MSE: {all_results[channel]['mean_mse']:.6f} "
            f"+/- {all_results[channel]['std_mse']:.6f}"
        )
    print(f"  Matched pairs: {all_results[channels[0]]['num_matched_pairs']}")
    print("=" * 60)

    results = {
        "timestamp": datetime.now().isoformat(),
        "generation_mode": generation_mode,
        "configuration": config,
        "metrics": all_results,
    }

    output_path = args.output_json or str(images_dir / "mse_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
