#!/usr/bin/env python3
"""
Stage 5: Intra-Nuclear Proportion Evaluation

Compute intra-nuclear protein proportion for true and generated images:
    proportion = sum(nuclear_mask * protein_image) / sum(protein_image)

This script is intended for protein_only generation mode where protein is generated
and landmark channels are conditioning inputs.

Usage:
    python evaluations/ca_esm/evaluate_intra_nuclear_prop.py \
        --images_dir /path/to/output_from_stage1 \
        --output_json /path/to/intra_nuclear_prop_results.json
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


def extract_base_name(path_str, prefix, channel):
    """Extract sample base name from path like true_<base>_<channel>.png."""
    stem = Path(path_str).stem
    if stem.startswith(f"{prefix}_"):
        stem = stem[len(prefix) + 1 :]
    suffix = f"_{channel}"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def load_channel_map(images_dir, prefix, channel):
    """
    Build mapping: sample base name -> image path for one channel.
    """
    images_dir = Path(images_dir)
    pattern = str(images_dir / f"{prefix}_*_{channel}.png")
    image_files = sorted(glob.glob(pattern))
    if len(image_files) == 0:
        raise FileNotFoundError(
            f"No files matching pattern '{prefix}_*_{channel}.png' in {images_dir}"
        )

    result = {}
    for path in image_files:
        base = extract_base_name(path, prefix=prefix, channel=channel)
        result[base] = path
    return result


def compute_otsu_mask(nucleus_img):
    """Fallback binary nucleus mask via Otsu threshold."""
    from skimage.filters import threshold_otsu

    threshold = threshold_otsu(nucleus_img)
    return nucleus_img > threshold


class StardistMasker:
    """Lazy loader for StarDist nucleus segmentation."""

    def __init__(self, model_name="2D_versatile_fluo", gaussian_sigma=1.0):
        self.model_name = model_name
        self.gaussian_sigma = gaussian_sigma
        self.model = None
        self._loaded = False
        self._error = None

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            from stardist.models import StarDist2D

            self._normalize = __import__("csbdeep.utils", fromlist=["normalize"]).normalize
            self._gaussian = __import__("skimage.filters", fromlist=["gaussian"]).gaussian
            self.model = StarDist2D.from_pretrained(self.model_name)
        except Exception as exc:
            self._error = exc
            self.model = None

    def available(self):
        self._load()
        return self.model is not None

    def predict_mask(self, nucleus_img):
        self._load()
        if self.model is None:
            raise RuntimeError(
                f"StarDist is unavailable: {type(self._error).__name__}: {self._error}"
            )
        processed = self._gaussian(self._normalize(nucleus_img), sigma=self.gaussian_sigma)
        labeled_nuclei, _ = self.model.predict_instances(processed)
        return labeled_nuclei > 0


def safe_prop(mask, protein_img, eps):
    """Compute masked protein proportion with denominator guard."""
    total = float(np.sum(protein_img))
    if total <= eps:
        return np.nan, total
    inside = float(np.sum(mask * protein_img))
    return inside / total, total


def safe_correlations(x, y):
    """Compute Pearson and Spearman on valid pairs; return NaN if undefined."""
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2:
        return np.nan, np.nan
    xv = x[valid]
    yv = y[valid]
    try:
        from scipy.stats import pearsonr, spearmanr

        pearson_val, _ = pearsonr(xv, yv)
        spearman_val, _ = spearmanr(xv, yv)
        return float(pearson_val), float(spearman_val)
    except Exception:
        # Pearson fallback without scipy
        x_centered = xv - np.mean(xv)
        y_centered = yv - np.mean(yv)
        denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
        pearson_val = np.nan if denom == 0 else float(np.sum(x_centered * y_centered) / denom)
        return pearson_val, np.nan


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute intra-nuclear proportion on pre-generated images"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing images_true/, images_gen/, and generation_metadata.json",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON path (default: {images_dir}/intra_nuclear_prop_results.json)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Small value used for zero-denominator guard (default: 1e-8)",
    )
    parser.add_argument(
        "--mask_method",
        type=str,
        default="stardist",
        choices=["stardist", "otsu"],
        help="Method to create nuclear masks (default: stardist)",
    )
    parser.add_argument(
        "--stardist_model",
        type=str,
        default="2D_versatile_fluo",
        help="StarDist pretrained model name (default: 2D_versatile_fluo)",
    )
    parser.add_argument(
        "--gaussian_sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma before StarDist prediction (default: 1.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Stage 5: Intra-Nuclear Proportion Evaluation")
    print("=" * 60)

    images_dir = Path(args.images_dir)
    images_dir_true = images_dir / "images_true"
    images_dir_gen = images_dir / "images_gen"
    metadata_path = images_dir / "generation_metadata.json"

    if not images_dir_true.exists() or not images_dir_gen.exists():
        raise FileNotFoundError(
            f"images_true/ or images_gen/ not found in {images_dir}. Run generate_images.py first."
        )

    # Load metadata if available.
    config = {}
    generation_mode = "protein_only"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        config = metadata.get("configuration", {})
        generation_mode = metadata.get("generation_mode", "protein_only")

    print(f"Generation mode: {generation_mode}")
    if generation_mode == "all_channels":
        print(
            "Skipping intra-nuclear proportion evaluation: "
            "metric is configured for protein_only mode."
        )
        sys.exit(0)

    true_nucleus = load_channel_map(images_dir_true, prefix="true", channel="nucleus")
    true_protein = load_channel_map(images_dir_true, prefix="true", channel="protein")
    gen_protein = load_channel_map(images_dir_gen, prefix="gen", channel="protein")

    matched_files = sorted(set(true_nucleus) & set(true_protein) & set(gen_protein))
    if len(matched_files) == 0:
        raise ValueError(
            "No matched basenames across true nucleus/protein and generated protein images."
        )
    print(f"Matched {len(matched_files)} samples")

    stardist_masker = StardistMasker(
        model_name=args.stardist_model, gaussian_sigma=args.gaussian_sigma
    )
    mask_method_requested = args.mask_method
    using_stardist = args.mask_method == "stardist" and stardist_masker.available()
    fallback_to_otsu = args.mask_method == "stardist" and not using_stardist
    if fallback_to_otsu:
        print(
            "WARNING: StarDist unavailable; falling back to Otsu masks. "
            f"Reason: {type(stardist_masker._error).__name__}: {stardist_masker._error}"
        )

    prop_true_vals = []
    prop_gen_vals = []
    mask_fraction_vals = []
    per_image = {}

    n_invalid_true = 0
    n_invalid_gen = 0
    n_stardist_failures = 0

    for name in tqdm(matched_files, desc="Computing intra-nuclear proportions"):
        nuc_img = load_png_image(true_nucleus[name])
        true_img = load_png_image(true_protein[name])
        gen_img = load_png_image(gen_protein[name])

        if using_stardist:
            try:
                mask = stardist_masker.predict_mask(nuc_img)
                mask_method_used = "stardist"
            except Exception:
                n_stardist_failures += 1
                mask = compute_otsu_mask(nuc_img)
                mask_method_used = "otsu_fallback"
        else:
            mask = compute_otsu_mask(nuc_img)
            mask_method_used = "otsu"

        prop_true, denom_true = safe_prop(mask, true_img, args.eps)
        prop_gen, denom_gen = safe_prop(mask, gen_img, args.eps)

        invalid_true = not np.isfinite(prop_true)
        invalid_gen = not np.isfinite(prop_gen)
        if invalid_true:
            n_invalid_true += 1
        if invalid_gen:
            n_invalid_gen += 1

        abs_error = np.nan
        sq_error = np.nan
        if np.isfinite(prop_true) and np.isfinite(prop_gen):
            abs_error = abs(prop_gen - prop_true)
            sq_error = (prop_gen - prop_true) ** 2

        mask_fraction = float(np.mean(mask))
        mask_fraction_vals.append(mask_fraction)
        prop_true_vals.append(prop_true)
        prop_gen_vals.append(prop_gen)
        per_image[name] = {
            "intra_nuclear_prop_true": float(prop_true) if np.isfinite(prop_true) else None,
            "intra_nuclear_prop_gen": float(prop_gen) if np.isfinite(prop_gen) else None,
            "abs_error": float(abs_error) if np.isfinite(abs_error) else None,
            "squared_error": float(sq_error) if np.isfinite(sq_error) else None,
            "protein_sum_true": denom_true,
            "protein_sum_gen": denom_gen,
            "nuclear_mask_fraction": mask_fraction,
            "mask_method_used": mask_method_used,
        }

    prop_true_arr = np.asarray(prop_true_vals, dtype=np.float64)
    prop_gen_arr = np.asarray(prop_gen_vals, dtype=np.float64)
    valid = np.isfinite(prop_true_arr) & np.isfinite(prop_gen_arr)

    if np.any(valid):
        diff = prop_gen_arr[valid] - prop_true_arr[valid]
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        bias = float(np.mean(diff))
        mean_true = float(np.mean(prop_true_arr[valid]))
        std_true = float(np.std(prop_true_arr[valid]))
        mean_gen = float(np.mean(prop_gen_arr[valid]))
        std_gen = float(np.std(prop_gen_arr[valid]))
    else:
        mae = np.nan
        rmse = np.nan
        bias = np.nan
        mean_true = np.nan
        std_true = np.nan
        mean_gen = np.nan
        std_gen = np.nan

    pearson_corr, spearman_corr = safe_correlations(prop_true_arr, prop_gen_arr)
    metrics = {
        "mask_method_requested": mask_method_requested,
        "mask_method_global": "stardist" if using_stardist else "otsu",
        "num_samples_matched": len(matched_files),
        "num_valid_pairs": int(np.sum(valid)),
        "num_invalid_true": n_invalid_true,
        "num_invalid_gen": n_invalid_gen,
        "num_stardist_failures": n_stardist_failures,
        "mean_intra_nuclear_prop_true": mean_true if np.isfinite(mean_true) else None,
        "std_intra_nuclear_prop_true": std_true if np.isfinite(std_true) else None,
        "mean_intra_nuclear_prop_gen": mean_gen if np.isfinite(mean_gen) else None,
        "std_intra_nuclear_prop_gen": std_gen if np.isfinite(std_gen) else None,
        "mae_intra_nuclear_prop": mae if np.isfinite(mae) else None,
        "rmse_intra_nuclear_prop": rmse if np.isfinite(rmse) else None,
        "bias_gen_minus_true": bias if np.isfinite(bias) else None,
        "pearson_correlation": pearson_corr if np.isfinite(pearson_corr) else None,
        "spearman_correlation": spearman_corr if np.isfinite(spearman_corr) else None,
        "mean_nuclear_mask_fraction": float(np.mean(mask_fraction_vals)),
        "std_nuclear_mask_fraction": float(np.std(mask_fraction_vals)),
    }

    print("\n" + "=" * 60)
    print("INTRA-NUCLEAR PROPORTION RESULTS")
    print("=" * 60)
    print(f"Matched pairs:                 {metrics['num_samples_matched']}")
    print(f"Valid pairs:                   {metrics['num_valid_pairs']}")
    print(f"Mask method (requested/used):  {metrics['mask_method_requested']} / {metrics['mask_method_global']}")
    print(f"Mean proportion (true):        {metrics['mean_intra_nuclear_prop_true']}")
    print(f"Mean proportion (gen):         {metrics['mean_intra_nuclear_prop_gen']}")
    print(f"MAE:                           {metrics['mae_intra_nuclear_prop']}")
    print(f"RMSE:                          {metrics['rmse_intra_nuclear_prop']}")
    print(f"Pearson corr:                  {metrics['pearson_correlation']}")
    print(f"Spearman corr:                 {metrics['spearman_correlation']}")
    print("=" * 60)

    results = {
        "timestamp": datetime.now().isoformat(),
        "generation_mode": generation_mode,
        "configuration": config,
        "metrics": metrics,
        "per_image": per_image,
    }

    output_path = args.output_json or str(images_dir / "intra_nuclear_prop_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
