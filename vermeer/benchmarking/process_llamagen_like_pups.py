#!/usr/bin/env python3
"""
Process CA_LlamaGen output to match PUPS single-cell crop format.

LlamaGen generates 256x256 FOV images (8x downsample from 2048x2048).
PUPS benchmarks use 512x512 FOVs (4x downsample) with 128x128 single-cell crops.
This script extracts 64x64 single-cell crops from the 256x256 FOVs (proportionally
equivalent to PUPS's 128x128 crops at 512x512) and upsamples them to 128x128
using cv2 INTER_CUBIC interpolation.

Usage:
    python benchmarking/process_llamagen_like_pups.py \
        --input_dir /path/to/evaluation_output \
        --num_workers 4
"""

import argparse
import os
import re
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.transform import resize as skimage_resize
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.exposure import rescale_intensity
from scipy.ndimage import center_of_mass


CHANNEL_NAMES = ['nucleus', 'microtubule', 'er', 'protein']

# Segmentation parameters scaled from PUPS (512x512) to LlamaGen (256x256).
# Area scales by (256/512)^2 = 0.25, lengths by 0.5.
MIN_NUC_SIZE = 25       # PUPS: 100
SIGMA = 2.5             # PUPS: 5.0
HOLE_AREA_THRESH = 75   # PUPS: 300
CROP_SIZE = 64          # PUPS: 128  (half, matching 256/512 ratio)
OUTPUT_SIZE = 128       # Upsample to match PUPS crop size


def load_fov_channels(image_dir, prefix, base_name):
    """Load 4-channel FOV from individual channel PNGs.

    Returns dict mapping channel_name -> (H, W) float32 array in [0, 1].
    Returns None if any channel is missing.
    """
    channels = {}
    for ch_name in CHANNEL_NAMES:
        path = os.path.join(image_dir, f"{prefix}_{base_name}_{ch_name}.png")
        if not os.path.exists(path):
            return None
        img = np.array(Image.open(path)).astype(np.float32) / 255.0
        channels[ch_name] = img
    return channels


def segment_nuclei(nucleus_img):
    """Segment nuclei from a 256x256 nucleus channel image.

    Follows the same pipeline as PUPS preprocess_fov but with parameters
    scaled for 256x256 (half of PUPS's 512x512).

    Returns labeled array where each nucleus has a unique integer label.
    """
    val = threshold_otsu(nucleus_img)
    smoothed = gaussian(nucleus_img, sigma=SIGMA)
    binary = smoothed > val
    binary = remove_small_holes(binary, area_threshold=HOLE_AREA_THRESH)
    labeled = label(binary)
    labeled = clear_border(labeled)
    labeled = remove_small_objects(labeled, min_size=MIN_NUC_SIZE)
    return labeled


def extract_crop_coordinates(labeled_nuclei):
    """Get (y, x) crop centers and bounding checks for each nucleus.

    Returns list of (crop_idx, y_start, y_end, x_start, x_end) tuples.
    Only includes crops fully within the image.
    """
    h, w = labeled_nuclei.shape
    n_nuclei = np.max(labeled_nuclei)
    half = CROP_SIZE // 2

    coords = []
    crop_idx = 0
    for i in range(1, n_nuclei + 1):
        mask = labeled_nuclei == i
        if np.sum(mask) <= MIN_NUC_SIZE:
            continue

        cy, cx = center_of_mass(mask)
        cx, cy = int(cx), int(cy)

        y1 = cy - half
        y2 = cy + half
        x1 = cx - half
        x2 = cx + half

        if y1 < 0 or x1 < 0 or y2 > h or x2 > w:
            continue

        coords.append((crop_idx, y1, y2, x1, x2))
        crop_idx += 1

    return coords


def crop_and_upsample(channel_img, y1, y2, x1, x2):
    """Extract a crop and upsample to OUTPUT_SIZE x OUTPUT_SIZE.

    Steps:
        1. Extract CROP_SIZE x CROP_SIZE region
        2. Rescale intensity to [0, 1]
        3. Upsample to OUTPUT_SIZE x OUTPUT_SIZE with cv2 INTER_CUBIC
    """
    crop = channel_img[y1:y2, x1:x2].astype(np.float32)
    crop = rescale_intensity(crop, out_range=(0.0, 1.0))
    upsampled = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE),
                           interpolation=cv2.INTER_CUBIC)
    # Clip to [0, 1] since cubic interpolation can overshoot
    upsampled = np.clip(upsampled, 0.0, 1.0)
    return upsampled


def save_crop_png(img_float, path):
    """Save a [0, 1] float image as uint8 PNG."""
    img_uint8 = (img_float * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)


def process_single_fov(args):
    """Process a single FOV: segment, crop, upsample, save.

    Processes both true and gen images using the same crop coordinates
    (derived from the true nucleus channel).
    """
    base_name, input_dir, out_dir_true, out_dir_gen = args

    true_dir = os.path.join(input_dir, "images_true")
    gen_dir = os.path.join(input_dir, "images_gen")

    # Load channels
    true_channels = load_fov_channels(true_dir, "true", base_name)
    gen_channels = load_fov_channels(gen_dir, "gen", base_name)

    if true_channels is None:
        return base_name, 0, "missing true channels"
    if gen_channels is None:
        return base_name, 0, "missing gen channels"

    # Segment nuclei from true nucleus channel
    labeled_nuclei = segment_nuclei(true_channels['nucleus'])
    crop_coords = extract_crop_coordinates(labeled_nuclei)

    if len(crop_coords) == 0:
        return base_name, 0, "no valid crops"

    # Extract, upsample, and save crops for both true and gen
    for crop_idx, y1, y2, x1, x2 in crop_coords:
        crop_suffix = f"_crop{crop_idx:03d}"
        for ch_name in CHANNEL_NAMES:
            # True
            true_crop = crop_and_upsample(true_channels[ch_name], y1, y2, x1, x2)
            true_path = os.path.join(
                out_dir_true, f"true_{base_name}{crop_suffix}_{ch_name}.png"
            )
            save_crop_png(true_crop, true_path)

            # Generated
            gen_crop = crop_and_upsample(gen_channels[ch_name], y1, y2, x1, x2)
            gen_path = os.path.join(
                out_dir_gen, f"gen_{base_name}{crop_suffix}_{ch_name}.png"
            )
            save_crop_png(gen_crop, gen_path)

    return base_name, len(crop_coords), "ok"


def discover_base_names(images_true_dir):
    """Discover unique FOV base names from true nucleus PNGs."""
    pattern = os.path.join(images_true_dir, "true_*_nucleus.png")
    paths = sorted(glob.glob(pattern))

    base_names = []
    for p in paths:
        fname = os.path.basename(p)
        # Strip "true_" prefix and "_nucleus.png" suffix
        match = re.match(r"true_(.+)_nucleus\.png$", fname)
        if match:
            base_names.append(match.group(1))

    return base_names


def main():
    parser = argparse.ArgumentParser(
        description="Process CA_LlamaGen output to PUPS-like single-cell crops"
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

    # Output directories (match PUPS-like folder layout)
    out_root_dir = os.path.join(input_dir, "images_processed_like_pups")
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

    total_crops = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_fov, ta): ta[0] for ta in task_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing FOVs"):
            base_name, n_crops, status = future.result()
            if status == "ok":
                total_crops += n_crops
            else:
                skipped += 1

    print(f"\nDone. {total_crops} crops from {len(base_names) - skipped} FOVs "
          f"({skipped} skipped).")
    print(f"Output root: {out_root_dir}")
    print(f"True crops:  {out_dir_true}")
    print(f"Gen crops:   {out_dir_gen}")


if __name__ == "__main__":
    main()
