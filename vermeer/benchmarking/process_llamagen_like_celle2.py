#!/usr/bin/env python3
"""
Process CA_LlamaGen output to match CELLE-2 resolution pipeline.

CELLE-2 takes 2048x2048 images, resizes to 600x600, then center-crops to 256x256.
LlamaGen resizes 2048x2048 directly to 256x256.

To make resolutions comparable for downstream evals, this script:
  1. Upscales 256x256 -> 600x600 (bicubic interpolation)
  2. Center-crops 600x600 -> 256x256

This is applied to both true and generated images.

Usage:
    python benchmarking/process_llamagen_like_celle2.py \
        --input_dir /path/to/evaluation_output \
        --num_workers 4
"""

import argparse
import os
import glob
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


CHANNEL_NAMES = ['nucleus', 'microtubule', 'er', 'protein']
UPSAMPLE_SIZE = 600
CROP_SIZE = 256


def upscale_and_center_crop(img_float):
    """Upscale a 256x256 image to 600x600, then center-crop to 256x256.

    Args:
        img_float: (H, W) float32 array in [0, 1]

    Returns:
        (256, 256) float32 array in [0, 1]
    """
    # Upscale to 600x600 with bicubic interpolation
    upscaled = cv2.resize(img_float, (UPSAMPLE_SIZE, UPSAMPLE_SIZE),
                          interpolation=cv2.INTER_CUBIC)
    # Clip to [0, 1] since cubic interpolation can overshoot
    upscaled = np.clip(upscaled, 0.0, 1.0)

    # Center crop to 256x256
    h, w = upscaled.shape[:2]
    y_start = (h - CROP_SIZE) // 2
    x_start = (w - CROP_SIZE) // 2
    cropped = upscaled[y_start:y_start + CROP_SIZE, x_start:x_start + CROP_SIZE]

    return cropped


def save_png(img_float, path):
    """Save a [0, 1] float image as uint8 PNG."""
    img_uint8 = (img_float * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)


def process_single_fov(args):
    """Process a single FOV: upscale and center-crop all channels for true and gen."""
    base_name, input_dir, out_dir_true, out_dir_gen = args

    true_dir = os.path.join(input_dir, "images_true")
    gen_dir = os.path.join(input_dir, "images_gen")

    for ch_name in CHANNEL_NAMES:
        # True image
        true_path = os.path.join(true_dir, f"true_{base_name}_{ch_name}.png")
        if not os.path.exists(true_path):
            return base_name, "missing true channels"

        true_img = np.array(Image.open(true_path)).astype(np.float32) / 255.0
        true_processed = upscale_and_center_crop(true_img)
        save_png(true_processed, os.path.join(
            out_dir_true, f"true_{base_name}_{ch_name}.png"
        ))

        # Generated image
        gen_path = os.path.join(gen_dir, f"gen_{base_name}_{ch_name}.png")
        if not os.path.exists(gen_path):
            return base_name, "missing gen channels"

        gen_img = np.array(Image.open(gen_path)).astype(np.float32) / 255.0
        gen_processed = upscale_and_center_crop(gen_img)
        save_png(gen_processed, os.path.join(
            out_dir_gen, f"gen_{base_name}_{ch_name}.png"
        ))

    return base_name, "ok"


def discover_base_names(images_true_dir):
    """Discover unique FOV base names from true nucleus PNGs."""
    pattern = os.path.join(images_true_dir, "true_*_nucleus.png")
    paths = sorted(glob.glob(pattern))

    base_names = []
    for p in paths:
        fname = os.path.basename(p)
        match = re.match(r"true_(.+)_nucleus\.png$", fname)
        if match:
            base_names.append(match.group(1))

    return base_names


def main():
    parser = argparse.ArgumentParser(
        description="Process CA_LlamaGen output to match CELLE-2 resolution pipeline"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing images_gen/ and images_true/"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of parallel workers (default: 4)"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    images_true_dir = os.path.join(input_dir, "images_true")
    images_gen_dir = os.path.join(input_dir, "images_gen")

    if not os.path.isdir(images_true_dir):
        raise FileNotFoundError(f"images_true/ not found in {input_dir}")
    if not os.path.isdir(images_gen_dir):
        raise FileNotFoundError(f"images_gen/ not found in {input_dir}")

    # Output directories
    out_root_dir = os.path.join(input_dir, "images_processed_like_celle2")
    out_dir_true = os.path.join(out_root_dir, "images_true")
    out_dir_gen = os.path.join(out_root_dir, "images_gen")
    os.makedirs(out_root_dir, exist_ok=True)
    os.makedirs(out_dir_true, exist_ok=True)
    os.makedirs(out_dir_gen, exist_ok=True)

    # Discover FOVs
    base_names = discover_base_names(images_true_dir)
    print(f"Found {len(base_names)} FOVs in {images_true_dir}")

    if len(base_names) == 0:
        print("No FOVs found. Check that images_true/ contains true_*_nucleus.png files.")
        return

    # Process in parallel
    task_args = [
        (bn, input_dir, out_dir_true, out_dir_gen)
        for bn in base_names
    ]

    processed = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_fov, ta): ta[0] for ta in task_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing FOVs"):
            base_name, status = future.result()
            if status == "ok":
                processed += 1
            else:
                skipped += 1
                print(f"  Skipped {base_name}: {status}")

    print(f"\nDone. Processed {processed} FOVs ({skipped} skipped).")
    print(f"Output root: {out_root_dir}")
    print(f"True images: {out_dir_true}")
    print(f"Gen images:  {out_dir_gen}")


if __name__ == "__main__":
    main()
