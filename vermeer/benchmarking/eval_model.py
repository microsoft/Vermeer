#!/usr/bin/env python3
"""
Model-agnostic evaluation script for microscopy image generation.

Runs all evaluation metrics (FID, MSE, SubCell localization, intra-nuclear
proportion) on pre-generated images. Decoupled from any specific generation
model — just provide a directory with true and generated images.

Expected directory layout:
    images_dir/
        images_true/   true_{basename}_{channel}.png
        images_gen/    gen_{basename}_{channel}.png

Channels: nucleus, microtubule, er, protein

Usage:
    # Evaluate protein-only generation (default)
    python benchmarking/eval_model.py --images_dir /path/to/output

    # Evaluate all-channels generation (skip subcell + intra-nuclear)
    python benchmarking/eval_model.py --images_dir /path/to/output --generation_mode all_channels

    # Skip specific evals
    python benchmarking/eval_model.py --images_dir /path/to/output --skip_subcell --skip_intra_nuclear
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Add CA_LlamaGen root to path so evaluation modules are importable
# ---------------------------------------------------------------------------
CA_LLAMAGEN_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(CA_LLAMAGEN_DIR))

from evaluations.ca_esm import evaluate_fid as fid_module
from evaluations.ca_esm import evaluate_mse as mse_module
from evaluations.ca_esm import evaluate_intra_nuclear_prop as inp_module
from evaluations.ca_esm.evaluate_subcell import (
    build_localization_lookup,
    build_subcell_records,
    compute_average_precision_scores,
    compute_f1_scores,
    compute_multilabel_ranking_metrics,
    compute_spearman_correlation,
    compute_top_k_multi_label_accuracy,
    convert_hpa_labels_to_binary,
    create_subcell_csv,
    get_localization_from_filename,
    get_subcell_class_names,
    parse_subcell_probabilities,
    parse_subcell_results,
    run_subcell_portable,
)


ALL_CHANNELS = ["protein", "nucleus", "microtubule", "er"]

import re

def strip_crop_suffix(filename):
    """Strip PUPS single-cell crop suffix (e.g. '_crop0') from a filename."""
    return re.sub(r"_crop\d+$", "", filename)


# ---------------------------------------------------------------------------
# Utility: auto-detect channels present in a directory
# ---------------------------------------------------------------------------

def detect_channels(images_dir):
    """Detect which channels are present by scanning filenames."""
    found = set()
    for ch in ALL_CHANNELS:
        pattern = str(Path(images_dir) / f"*_{ch}.png")
        if glob.glob(pattern):
            found.add(ch)
    return sorted(found, key=ALL_CHANNELS.index)


def detect_filenames(images_dir, prefix="true"):
    """Extract unique basenames from images in a directory.

    Scans for ``{prefix}_{basename}_protein.png`` first, then falls back to
    any channel present.
    """
    images_dir = Path(images_dir)
    basenames = set()

    for ch in ALL_CHANNELS:
        for path in images_dir.glob(f"{prefix}_*_{ch}.png"):
            stem = path.stem
            # Strip prefix
            if stem.startswith(f"{prefix}_"):
                stem = stem[len(prefix) + 1:]
            # Strip channel suffix
            suffix = f"_{ch}"
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
            basenames.add(stem)

    return sorted(basenames)


# ---------------------------------------------------------------------------
# Stage: FID
# ---------------------------------------------------------------------------

def run_fid(images_dir_true, images_dir_gen, channels, batch_size, device):
    """Compute per-channel FID scores."""
    results = {}
    for channel in channels:
        print(f"\n  Channel: {channel.upper()}")
        true_images, true_fnames = fid_module.load_images_from_directory(
            str(images_dir_true), channel=channel
        )
        gen_images, gen_fnames = fid_module.load_images_from_directory(
            str(images_dir_gen), channel=channel
        )
        true_idx, gen_idx, common = fid_module.match_image_pairs(true_fnames, gen_fnames)
        print(f"  Matched {len(common)} pairs")

        fid_score = fid_module.compute_fid_score(
            true_images[true_idx], gen_images[gen_idx],
            batch_size=batch_size, device=device,
        )
        results[channel] = {
            "fid_score": fid_score,
            "num_matched_pairs": len(common),
            "num_true_images": len(true_fnames),
            "num_gen_images": len(gen_fnames),
        }
        print(f"  {channel} FID: {fid_score:.4f}")

    return results


# ---------------------------------------------------------------------------
# Stage: MSE
# ---------------------------------------------------------------------------

def run_mse(images_dir_true, images_dir_gen, channels):
    """Compute per-channel MSE scores."""
    results = {}
    for channel in channels:
        print(f"\n  Channel: {channel.upper()}")
        true_images, true_fnames = mse_module.load_images_from_directory(
            str(images_dir_true), channel=channel
        )
        gen_images, gen_fnames = mse_module.load_images_from_directory(
            str(images_dir_gen), channel=channel
        )
        true_idx, gen_idx, matched = mse_module.match_image_pairs(true_fnames, gen_fnames)
        print(f"  Matched {len(matched)} pairs")

        per_image_mse = mse_module.compute_per_image_mse(
            true_images[true_idx], gen_images[gen_idx]
        )
        per_image_metrics = {
            fname: float(val) for fname, val in zip(matched, per_image_mse)
        }
        results[channel] = {
            "mean_mse": float(np.mean(per_image_mse)),
            "std_mse": float(np.std(per_image_mse)),
            "num_matched_pairs": len(matched),
            "num_true_images": len(true_fnames),
            "num_gen_images": len(gen_fnames),
            "per_image_mse": per_image_metrics,
        }
        print(
            f"  {channel} MSE: {results[channel]['mean_mse']:.6f} "
            f"+/- {results[channel]['std_mse']:.6f}"
        )

    return results


# ---------------------------------------------------------------------------
# Stage: SubCell localization
# ---------------------------------------------------------------------------

def run_subcell(images_dir_true, images_dir_gen, filenames, hpa_csv, subcell_dir, output_dir, min_examples, f1_threshold=0.15):
    """Run SubCellPortable and compute localization metrics."""
    import pandas as pd

    # Build SubCell CSV records
    print("  Building SubCellPortable CSV records...")
    records_true = build_subcell_records(str(images_dir_true), filenames, "true")
    records_gen = build_subcell_records(str(images_dir_gen), filenames, "gen")

    csv_true = str(output_dir / "path_list_true.csv")
    csv_gen = str(output_dir / "path_list_gen.csv")
    create_subcell_csv(records_true, csv_true)
    create_subcell_csv(records_gen, csv_gen)

    results_dir_true = output_dir / "results_true"
    results_dir_gen = output_dir / "results_gen"
    results_dir_true.mkdir(parents=True, exist_ok=True)
    results_dir_gen.mkdir(parents=True, exist_ok=True)

    # Run SubCellPortable
    print("  Running SubCellPortable on true images...")
    result_csv_true = run_subcell_portable(
        Path(csv_true).absolute(), results_dir_true.absolute(), subcell_dir
    )
    torch.cuda.empty_cache()

    print("  Running SubCellPortable on generated images...")
    result_csv_gen = run_subcell_portable(
        Path(csv_gen).absolute(), results_dir_gen.absolute(), subcell_dir
    )
    torch.cuda.empty_cache()

    # Parse results
    subcell_true_dict = parse_subcell_results(result_csv_true)
    subcell_gen_dict = parse_subcell_results(result_csv_gen)
    probs_true = parse_subcell_probabilities(result_csv_true)
    probs_gen = parse_subcell_probabilities(result_csv_gen)

    # Load HPA annotations
    print(f"  Loading HPA annotations from {hpa_csv}...")
    hpa_df = pd.read_csv(hpa_csv)
    hpa_lookup = build_localization_lookup(hpa_df)

    hpa_labels = []
    subcell_true_list = []
    subcell_gen_list = []
    for filename in filenames:
        lookup_name = strip_crop_suffix(filename)
        hpa_loc = get_localization_from_filename(hpa_lookup, lookup_name + ".npy")
        hpa_labels.append(hpa_loc)
        subcell_true_list.append(subcell_true_dict.get(f"true_{filename}", []))
        subcell_gen_list.append(subcell_gen_dict.get(f"gen_{filename}", []))

    class_names = get_subcell_class_names()
    y_true = convert_hpa_labels_to_binary(hpa_labels, filenames, class_names)

    # Compute metrics
    metrics = {}

    metrics["top_3_hpa_vs_subcell_true"] = compute_top_k_multi_label_accuracy(
        hpa_labels, subcell_true_list, k=3
    )
    metrics["top_3_hpa_vs_subcell_gen"] = compute_top_k_multi_label_accuracy(
        hpa_labels, subcell_gen_list, k=3
    )
    metrics["top_1_hpa_vs_subcell_true"] = compute_top_k_multi_label_accuracy(
        hpa_labels, subcell_true_list, k=1
    )
    metrics["top_1_hpa_vs_subcell_gen"] = compute_top_k_multi_label_accuracy(
        hpa_labels, subcell_gen_list, k=1
    )

    spearman_correlations = compute_spearman_correlation(probs_true, probs_gen, filenames)
    avg_spearman = np.mean(list(spearman_correlations.values())) if spearman_correlations else -1.0
    metrics["avg_spearman_correlation"] = float(avg_spearman)
    metrics["spearman_correlations_per_sample"] = spearman_correlations

    mlrap_true = compute_multilabel_ranking_metrics(
        hpa_labels, probs_true, filenames, prefix="true", y_true=y_true
    )
    metrics["hpa_vs_subcell_true_mlrap"] = mlrap_true["mlrap"]
    metrics["hpa_vs_subcell_true_coverage_error"] = mlrap_true["coverage_error"]

    mlrap_gen = compute_multilabel_ranking_metrics(
        hpa_labels, probs_gen, filenames, prefix="gen", y_true=y_true
    )
    metrics["hpa_vs_subcell_gen_mlrap"] = mlrap_gen["mlrap"]
    metrics["hpa_vs_subcell_gen_coverage_error"] = mlrap_gen["coverage_error"]

    ap_true = compute_average_precision_scores(
        hpa_labels, probs_true, filenames, prefix="true", y_true=y_true, min_examples=min_examples
    )
    metrics["hpa_vs_subcell_true_macro_ap"] = ap_true["macro_ap"]
    metrics["hpa_vs_subcell_true_micro_ap"] = ap_true["micro_ap"]

    ap_gen = compute_average_precision_scores(
        hpa_labels, probs_gen, filenames, prefix="gen", y_true=y_true, min_examples=min_examples
    )
    metrics["hpa_vs_subcell_gen_macro_ap"] = ap_gen["macro_ap"]
    metrics["hpa_vs_subcell_gen_micro_ap"] = ap_gen["micro_ap"]

    f1_true = compute_f1_scores(
        hpa_labels, probs_true, filenames, prefix="true", threshold=f1_threshold, y_true=y_true
    )
    metrics["hpa_vs_subcell_true_macro_f1"] = f1_true["macro_f1"]
    metrics["hpa_vs_subcell_true_micro_f1"] = f1_true["micro_f1"]

    f1_gen = compute_f1_scores(
        hpa_labels, probs_gen, filenames, prefix="gen", threshold=f1_threshold, y_true=y_true
    )
    metrics["hpa_vs_subcell_gen_macro_f1"] = f1_gen["macro_f1"]
    metrics["hpa_vs_subcell_gen_micro_f1"] = f1_gen["micro_f1"]

    return metrics


# ---------------------------------------------------------------------------
# Stage: Intra-nuclear proportion
# ---------------------------------------------------------------------------

def run_intra_nuclear_prop(images_dir_true, images_dir_gen, mask_method, stardist_model="2D_versatile_fluo"):
    """Compute intra-nuclear proportion metrics."""
    eps = 1e-8

    true_nucleus = inp_module.load_channel_map(str(images_dir_true), prefix="true", channel="nucleus")
    true_protein = inp_module.load_channel_map(str(images_dir_true), prefix="true", channel="protein")
    gen_protein = inp_module.load_channel_map(str(images_dir_gen), prefix="gen", channel="protein")

    matched = sorted(set(true_nucleus) & set(true_protein) & set(gen_protein))
    if len(matched) == 0:
        raise ValueError("No matched samples for intra-nuclear proportion")
    print(f"  Matched {len(matched)} samples")

    stardist_masker = inp_module.StardistMasker(model_name=stardist_model)
    using_stardist = mask_method == "stardist" and stardist_masker.available()
    if mask_method == "stardist" and not using_stardist:
        print("  WARNING: StarDist unavailable, falling back to Otsu masks")

    from tqdm import tqdm

    prop_true_vals = []
    prop_gen_vals = []
    mask_fraction_vals = []
    per_image = {}
    n_stardist_failures = 0

    for name in tqdm(matched, desc="  Intra-nuclear prop"):
        nuc_img = inp_module.load_png_image(true_nucleus[name])
        true_prot = inp_module.load_png_image(true_protein[name])
        gen_prot = inp_module.load_png_image(gen_protein[name])

        if using_stardist:
            try:
                mask = stardist_masker.predict_mask(nuc_img)
            except Exception:
                n_stardist_failures += 1
                mask = inp_module.compute_otsu_mask(nuc_img)
        else:
            mask = inp_module.compute_otsu_mask(nuc_img)

        prop_true, denom_true = inp_module.safe_prop(mask, true_prot, eps)
        prop_gen, denom_gen = inp_module.safe_prop(mask, gen_prot, eps)

        mask_frac = float(np.mean(mask))
        mask_fraction_vals.append(mask_frac)
        prop_true_vals.append(prop_true)
        prop_gen_vals.append(prop_gen)

        abs_err = abs(prop_gen - prop_true) if np.isfinite(prop_true) and np.isfinite(prop_gen) else np.nan
        per_image[name] = {
            "intra_nuclear_prop_true": float(prop_true) if np.isfinite(prop_true) else None,
            "intra_nuclear_prop_gen": float(prop_gen) if np.isfinite(prop_gen) else None,
            "abs_error": float(abs_err) if np.isfinite(abs_err) else None,
            "nuclear_mask_fraction": mask_frac,
        }

    prop_true_arr = np.asarray(prop_true_vals, dtype=np.float64)
    prop_gen_arr = np.asarray(prop_gen_vals, dtype=np.float64)
    valid = np.isfinite(prop_true_arr) & np.isfinite(prop_gen_arr)

    if np.any(valid):
        diff = prop_gen_arr[valid] - prop_true_arr[valid]
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
    else:
        mae = float("nan")
        rmse = float("nan")

    pearson_corr, spearman_corr = inp_module.safe_correlations(prop_true_arr, prop_gen_arr)

    metrics = {
        "mask_method": "stardist" if using_stardist else "otsu",
        "num_samples_matched": len(matched),
        "num_valid_pairs": int(np.sum(valid)),
        "num_stardist_failures": n_stardist_failures,
        "mae_intra_nuclear_prop": mae if np.isfinite(mae) else None,
        "rmse_intra_nuclear_prop": rmse if np.isfinite(rmse) else None,
        "pearson_correlation": pearson_corr if np.isfinite(pearson_corr) else None,
        "spearman_correlation": spearman_corr if np.isfinite(spearman_corr) else None,
        "mean_nuclear_mask_fraction": float(np.mean(mask_fraction_vals)),
    }

    return metrics, per_image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Model-agnostic evaluation for microscopy image generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected directory layout:
    images_dir/
        images_true/   true_{basename}_{channel}.png
        images_gen/    gen_{basename}_{channel}.png

Examples:
    # Evaluate protein-only generation
    python benchmarking/eval_model.py --images_dir /path/to/output

    # Evaluate all-channels, skip subcell
    python benchmarking/eval_model.py --images_dir /path/to/output \\
        --generation_mode all_channels

    # Only run FID and MSE
    python benchmarking/eval_model.py --images_dir /path/to/output \\
        --skip_subcell --skip_intra_nuclear
""",
    )

    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing images_true/ and images_gen/")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (default: images_dir)")
    parser.add_argument("--generation_mode", type=str, default="protein_only",
                        choices=["protein_only", "all_channels"],
                        help="Generation mode: protein_only runs all evals, "
                             "all_channels skips subcell and intra-nuclear (default: protein_only)")
    parser.add_argument("--channels", type=str, nargs="+", default=None,
                        choices=ALL_CHANNELS,
                        help="Channels to evaluate for FID/MSE (default: auto-detect)")
    parser.add_argument("--hpa_csv", type=str,
                        default="/n/home08/skambhampati/ar-microscopy-gen/code/data_preparation/hpa_cell-line-download.csv",  # change to your path
                        help="Path to HPA annotations CSV")
    parser.add_argument("--subcell_dir", type=str,
                        default=os.path.expanduser("~/SubCellPortable"),
                        help="Path to SubCellPortable directory")
    parser.add_argument("--fid_batch_size", type=int, default=32,
                        help="Batch size for FID computation")
    parser.add_argument("--mask_method", type=str, default="stardist",
                        choices=["stardist", "otsu"],
                        help="Nuclear mask method for intra-nuclear proportion")
    parser.add_argument("--min_examples", type=int, default=10,
                        help="Minimum number of examples for each class for SubCell localization average precision calculation")
    parser.add_argument("--f1_threshold", type=float, default=0.15,
                        help="Probability threshold for F1 score binarization (default: 0.15)")

    # Skip flags
    parser.add_argument("--skip_subcell", action="store_true",
                        help="Skip SubCell localization evaluation")
    parser.add_argument("--skip_fid", action="store_true",
                        help="Skip FID computation")
    parser.add_argument("--skip_mse", action="store_true",
                        help="Skip MSE computation")
    parser.add_argument("--skip_intra_nuclear", action="store_true",
                        help="Skip intra-nuclear proportion evaluation")

    return parser.parse_args()


def main():
    args = parse_args()

    images_dir = Path(args.images_dir)
    images_dir_true = images_dir / "images_true"
    images_dir_gen = images_dir / "images_gen"
    output_dir = Path(args.output_dir) if args.output_dir else images_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir_true.exists() or not images_dir_gen.exists():
        raise FileNotFoundError(
            f"Expected images_true/ and images_gen/ in {images_dir}"
        )

    # Auto-detect channels
    if args.channels:
        channels = args.channels
    elif args.generation_mode == "all_channels":
        channels = detect_channels(str(images_dir_gen))
        if not channels:
            channels = ALL_CHANNELS
    else:
        channels = ["protein"]

    # Detect sample filenames
    filenames = detect_filenames(str(images_dir_true), prefix="true")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Model-Agnostic Evaluation Pipeline")
    print("=" * 60)
    print(f"Images dir:        {images_dir}")
    print(f"Output dir:        {output_dir}")
    print(f"Generation mode:   {args.generation_mode}")
    print(f"Channels (FID/MSE):{channels}")
    print(f"Samples detected:  {len(filenames)}")
    print(f"Device:            {device}")
    print(f"Skip subcell:      {args.skip_subcell}")
    print(f"Skip FID:          {args.skip_fid}")
    print(f"Skip MSE:          {args.skip_mse}")
    print(f"Skip intra-nuclear:{args.skip_intra_nuclear}")
    print("=" * 60)

    combined = {}

    # --- FID ---
    if not args.skip_fid:
        print("\n" + "=" * 60)
        print("Stage: FID Evaluation")
        print("=" * 60)
        fid_results = run_fid(
            images_dir_true, images_dir_gen, channels, args.fid_batch_size, device
        )
        fid_data = {"timestamp": datetime.now().isoformat(), "metrics": fid_results}
        fid_path = output_dir / "fid_results.json"
        with open(fid_path, "w") as f:
            json.dump(fid_data, f, indent=2)
        print(f"  Saved to {fid_path}")
        combined["fid_results"] = fid_data
        torch.cuda.empty_cache()

    # --- MSE ---
    if not args.skip_mse:
        print("\n" + "=" * 60)
        print("Stage: MSE Evaluation")
        print("=" * 60)
        mse_results = run_mse(images_dir_true, images_dir_gen, channels)
        mse_data = {"timestamp": datetime.now().isoformat(), "metrics": mse_results}
        mse_path = output_dir / "mse_results.json"
        with open(mse_path, "w") as f:
            json.dump(mse_data, f, indent=2)
        print(f"  Saved to {mse_path}")
        combined["mse_results"] = mse_data

    # --- SubCell ---
    run_subcell_eval = (
        not args.skip_subcell
        and args.generation_mode == "protein_only"
    )
    if args.generation_mode != "protein_only" and not args.skip_subcell:
        print("\nSkipping SubCell evaluation (only available in protein_only mode)")

    if run_subcell_eval:
        print("\n" + "=" * 60)
        print("Stage: SubCell Localization Evaluation")
        print("=" * 60)
        subcell_metrics = run_subcell(
            images_dir_true, images_dir_gen, filenames,
            args.hpa_csv, args.subcell_dir, output_dir,
            args.min_examples, args.f1_threshold,
        )
        subcell_data = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(filenames),
            "metrics": subcell_metrics,
        }
        subcell_path = output_dir / "subcell_results.json"
        with open(subcell_path, "w") as f:
            json.dump(subcell_data, f, indent=2, default=str)
        print(f"  Saved to {subcell_path}")
        combined["subcell_results"] = subcell_data
        torch.cuda.empty_cache()

    # --- Intra-Nuclear Proportion ---
    run_inp = (
        not args.skip_intra_nuclear
        and args.generation_mode == "protein_only"
    )
    if args.generation_mode != "protein_only" and not args.skip_intra_nuclear:
        print("\nSkipping intra-nuclear proportion (only available in protein_only mode)")

    if run_inp:
        print("\n" + "=" * 60)
        print("Stage: Intra-Nuclear Proportion Evaluation")
        print("=" * 60)
        inp_metrics, inp_per_image = run_intra_nuclear_prop(
            images_dir_true, images_dir_gen, args.mask_method,
        )
        inp_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": inp_metrics,
            "per_image": inp_per_image,
        }
        inp_path = output_dir / "intra_nuclear_prop_results.json"
        with open(inp_path, "w") as f:
            json.dump(inp_data, f, indent=2)
        print(f"  Saved to {inp_path}")
        combined["intra_nuclear_prop_results"] = inp_data

    # --- Merge ---
    combined_path = output_dir / "eval_results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if "fid_results" in combined:
        print("--- FID ---")
        for ch, vals in combined["fid_results"]["metrics"].items():
            print(f"  {ch:12s} FID: {vals['fid_score']:.4f}  ({vals['num_matched_pairs']} pairs)")

    if "mse_results" in combined:
        print("--- MSE ---")
        for ch, vals in combined["mse_results"]["metrics"].items():
            print(f"  {ch:12s} MSE: {vals['mean_mse']:.6f} +/- {vals['std_mse']:.6f}")

    if "subcell_results" in combined:
        print("--- SubCell Localization ---")
        m = combined["subcell_results"]["metrics"]
        for key in [
            "top_1_hpa_vs_subcell_true", "top_1_hpa_vs_subcell_gen",
            "top_3_hpa_vs_subcell_true", "top_3_hpa_vs_subcell_gen",
            "avg_spearman_correlation",
            "hpa_vs_subcell_true_mlrap", "hpa_vs_subcell_gen_mlrap",
            "hpa_vs_subcell_true_macro_f1", "hpa_vs_subcell_gen_macro_f1",
        ]:
            val = m.get(key)
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")

    if "intra_nuclear_prop_results" in combined:
        print("--- Intra-Nuclear Proportion ---")
        m = combined["intra_nuclear_prop_results"]["metrics"]
        print(f"  MAE:          {m['mae_intra_nuclear_prop']}")
        print(f"  RMSE:         {m['rmse_intra_nuclear_prop']}")
        print(f"  Pearson:      {m['pearson_correlation']}")
        print(f"  Spearman:     {m['spearman_correlation']}")

    print("=" * 60)
    print(f"Combined results: {combined_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
