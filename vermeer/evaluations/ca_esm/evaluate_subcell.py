#!/usr/bin/env python3
"""
Stage 2: SubCell Localization Evaluation

Run SubCellPortable on pre-generated images and compute localization metrics.
Expects images from generate_images.py (Stage 1).

Usage:
    python evaluations/ca_esm/evaluate_subcell.py \
        --images_dir /path/to/output_from_stage1 \
        --output_json /path/to/subcell_results.json
"""

import numpy as np
from pathlib import Path
import argparse
import json
import importlib.util
import torch
import sys
import os
import csv
import ast
from datetime import datetime

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    label_ranking_average_precision_score,
    coverage_error,
    f1_score,
    average_precision_score,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


#################################################################################
#                          SubCellPortable Functions                             #
#################################################################################

def create_subcell_csv(records, csv_path):
    """Create SubCellPortable CSV file."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["r_image", "y_image", "b_image", "g_image", "output_prefix"]
        )
        writer.writeheader()
        writer.writerows(records)


def run_subcell_portable(csv_path, output_dir, subcell_dir):
    """Run SubCellPortable in-process via its Python entrypoint."""
    print(f"Running SubCellPortable on {csv_path}...")

    subcell_dir = os.path.abspath(os.path.expanduser(str(subcell_dir)))
    csv_path = str(csv_path)
    output_dir = str(output_dir)

    if not os.path.exists(subcell_dir):
        raise FileNotFoundError(f"SubCellPortable directory not found: {subcell_dir}")

    process_py = os.path.join(subcell_dir, "process.py")
    if not os.path.exists(process_py):
        raise FileNotFoundError(f"SubCellPortable process.py not found: {process_py}")

    config_yaml = os.path.join(subcell_dir, "config.yaml")
    cli_args = [
        "process.py",
        "--path_list", csv_path,
        "--output_dir", output_dir,
        "--model_channels", "rybg",
        "--create_csv",
    ]
    if os.path.exists(config_yaml):
        cli_args.extend(["--config", config_yaml])

    original_argv = sys.argv[:]
    original_cwd = os.getcwd()
    original_sys_path = sys.path[:]
    module_name = "_subcellportable_process"

    try:
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(subcell_dir)

        if subcell_dir not in sys.path:
            sys.path.insert(0, subcell_dir)

        module = sys.modules.get(module_name)
        if module is None or getattr(module, "__file__", None) != process_py:
            spec = importlib.util.spec_from_file_location(module_name, process_py)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load SubCellPortable module from {process_py}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

        sys.argv = cli_args
        module.run_inference()
        print("SubCellPortable completed successfully")
    except Exception as e:
        existing_module = sys.modules.get(module_name)
        if existing_module is not None and getattr(existing_module, "__file__", None) == process_py:
            if not hasattr(existing_module, "run_inference"):
                sys.modules.pop(module_name, None)
        print(f"ERROR: SubCellPortable failed: {type(e).__name__}: {e}")
        raise
    finally:
        sys.argv = original_argv
        os.chdir(original_cwd)
        sys.path = original_sys_path

    result_csv = os.path.join(output_dir, "result.csv")
    if not os.path.exists(result_csv):
        raise FileNotFoundError(f"SubCellPortable did not create result.csv at {result_csv}")

    return result_csv


def parse_subcell_results(result_csv_path, top_k=3):
    """Parse SubCellPortable results."""
    df = pd.read_csv(result_csv_path)
    results = {}
    for _, row in df.iterrows():
        sample_id = row.iloc[0]
        top_3_str = row["top_3_classes_names"]
        if pd.notna(top_3_str):
            top_3_labels = [label.strip() for label in str(top_3_str).split(",")]
        else:
            top_3_labels = []
        results[sample_id] = top_3_labels[:top_k]
    return results


def parse_subcell_probabilities(result_csv_path):
    """Parse SubCellPortable prediction probabilities."""
    df = pd.read_csv(result_csv_path)
    prob_cols = [col for col in df.columns if col.startswith("prob")]
    results = {}
    for _, row in df.iterrows():
        sample_id = row.iloc[0]
        probs = row[prob_cols].values.astype(float)
        results[sample_id] = probs
    return results


#################################################################################
#                          HPA Localization Functions                            #
#################################################################################

CLASS2NAME = {
    0: "Actin filaments",
    1: "Aggresome",
    2: "Cell Junctions",
    3: "Centriolar satellite",
    4: "Centrosome",
    5: "Cytokinetic bridge",
    6: "Cytoplasmic bodies",
    7: "Cytosol",
    8: "Endoplasmic reticulum",
    9: "Endosomes",
    10: "Focal adhesion sites",
    11: "Golgi apparatus",
    12: "Intermediate filaments",
    13: "Lipid droplets",
    14: "Lysosomes",
    15: "Microtubules",
    16: "Midbody",
    17: "Mitochondria",
    18: "Mitotic chromosome",
    19: "Mitotic spindle",
    20: "Nuclear bodies",
    21: "Nuclear membrane",
    22: "Nuclear speckles",
    23: "Nucleoli",
    24: "Nucleoli fibrillar center",
    25: "Nucleoli rim",
    26: "Nucleoplasm",
    27: "Peroxisomes",
    28: "Plasma membrane",
    29: "Vesicles",
    30: "Negative",
}


# Mapping from HPA label names to SubCellPortable class names where they differ.
# After normalize_label(), HPA labels that don't directly match a SubCell class
# are looked up here so they map to the correct column in y_true.
# HPA uses "No staining"; SubCellPortable uses "Negative" (prob30).
# HPA uses "Microtubule organizing center"; closest SubCellPortable class is "Centrosome".
HPA_TO_SUBCELL_LABEL = {
    "no staining": "negative",
    "microtubule organizing center": "centrosome",
}


def normalize_label(label):
    """Normalize a localization label for comparison.

    Applies lowercasing, whitespace collapsing, and the HPA → SubCell
    synonym mapping so that HPA annotations align with SubCellPortable
    class names.
    """
    if label is None:
        return None
    label = str(label).lower().replace("_", " ")
    label = " ".join(label.split())
    return HPA_TO_SUBCELL_LABEL.get(label, label)


def build_localization_lookup(df):
    """Pre-build a dictionary from (uniprot, antibody, cell_line, fov) -> localizations list."""
    lookup = {}
    for _, row in df.iterrows():
        key = (row["uniprot"], row["antibody"], row["cell_line"], int(row["fov"]))
        loc_str = row["localizations"]
        try:
            localizations = ast.literal_eval(loc_str)
            lookup[key] = localizations if localizations else None
        except Exception:
            lookup[key] = None
    return lookup


def get_localization_from_filename(lookup, filename):
    """O(1) lookup version."""
    parts = filename.replace(".npy", "").split("_")
    if len(parts) < 5 or parts[-2] != "fov":
        return None
    uniprot = parts[0]
    antibody = parts[1]
    fov = parts[-1]
    cell_line = "_".join(parts[2:-2])
    return lookup.get((uniprot, antibody, cell_line, int(fov)))


def get_subcell_class_names():
    """Return SubCellPortable class names in prob00–prob30 column order.

    This ordering matches SubCellPortable's CLASS2NAME dict exactly (alphabetical),
    so that class index i corresponds to column prob{i:02d} in result.csv.
    """
    return [
        "Actin filaments",           # prob00
        "Aggresome",                 # prob01
        "Cell Junctions",            # prob02
        "Centriolar satellite",      # prob03
        "Centrosome",                # prob04
        "Cytokinetic bridge",        # prob05
        "Cytoplasmic bodies",        # prob06
        "Cytosol",                   # prob07
        "Endoplasmic reticulum",     # prob08
        "Endosomes",                 # prob09
        "Focal adhesion sites",      # prob10
        "Golgi apparatus",           # prob11
        "Intermediate filaments",    # prob12
        "Lipid droplets",            # prob13
        "Lysosomes",                 # prob14
        "Microtubules",              # prob15
        "Midbody",                   # prob16
        "Mitochondria",              # prob17
        "Mitotic chromosome",        # prob18
        "Mitotic spindle",           # prob19
        "Nuclear bodies",            # prob20
        "Nuclear membrane",          # prob21
        "Nuclear speckles",          # prob22
        "Nucleoli",                  # prob23
        "Nucleoli fibrillar center", # prob24
        "Nucleoli rim",              # prob25
        "Nucleoplasm",               # prob26
        "Peroxisomes",               # prob27
        "Plasma membrane",           # prob28
        "Vesicles",                  # prob29
        "Negative",                  # prob30
    ]


def convert_hpa_labels_to_binary(hpa_labels_list, all_filenames, class_names):
    """Convert HPA labels to binary multi-label format.

    Indexes by ``class_names`` (SubCellPortable ordering) so that columns
    align with SubCellPortable probability outputs.
    """
    n_samples = len(all_filenames)
    n_classes = len(class_names)
    y_true = np.zeros((n_samples, n_classes), dtype=int)

    # Build index from the SubCellPortable class list so columns match y_scores
    name_to_idx = {normalize_label(name): idx for idx, name in enumerate(class_names)}

    for i, hpa_labels in enumerate(hpa_labels_list):
        if hpa_labels is not None and len(hpa_labels) > 0:
            for label in hpa_labels:
                normalized_label = normalize_label(label)
                if normalized_label in name_to_idx:
                    y_true[i, name_to_idx[normalized_label]] = 1
    return y_true


#################################################################################
#                              Metric Functions                                  #
#################################################################################

def compute_top_k_multi_label_accuracy(true_labels, pred_labels, k=3):
    n_valid = 0
    n_correct = 0
    for i in range(len(true_labels)):
        if true_labels[i] is not None and len(true_labels[i]) > 0 and pred_labels[i] is not None:
            n_valid += 1
            normalized_true = set(normalize_label(l) for l in true_labels[i] if l is not None)
            normalized_pred = [normalize_label(l) for l in pred_labels[i][:k] if l is not None]
            if any(p in normalized_true for p in normalized_pred):
                n_correct += 1
    return n_correct / n_valid if n_valid > 0 else 0.0


def compute_spearman_correlation(probs_true, probs_gen, all_filenames):
    """Compute Spearman correlation of predicted label rankings."""
    correlations = {}
    for filename in all_filenames:
        sample_id_true = f"true_{filename}"
        sample_id_gen = f"gen_{filename}"
        if sample_id_true in probs_true and sample_id_gen in probs_gen:
            scores_true = probs_true[sample_id_true]
            scores_gen = probs_gen[sample_id_gen]
            if np.any(np.isnan(scores_true)) or np.any(np.isnan(scores_gen)):
                continue
            corr, _ = spearmanr(scores_true, scores_gen)
            correlations[filename] = float(corr)
    return correlations


def compute_multilabel_ranking_metrics(hpa_labels_list, probs_dict, all_filenames, prefix="true", y_true=None):
    if y_true is None:
        y_true = convert_hpa_labels_to_binary(hpa_labels_list, all_filenames, get_subcell_class_names())

    y_scores = []
    valid_indices = []
    for i, filename in enumerate(all_filenames):
        sample_id = f"{prefix}_{filename}"
        if sample_id in probs_dict and hpa_labels_list[i] is not None and len(hpa_labels_list[i]) > 0:
            scores = probs_dict[sample_id]
            if np.any(np.isnan(scores)):
                continue
            y_scores.append(scores)
            valid_indices.append(i)

    if len(valid_indices) == 0:
        return {"mlrap": -1.0, "coverage_error": -1.0, "n_valid_samples": 0}

    y_true_valid = y_true[valid_indices]
    y_scores = np.array(y_scores)
    return {
        "mlrap": label_ranking_average_precision_score(y_true_valid, y_scores),
        "coverage_error": coverage_error(y_true_valid, y_scores),
        "n_valid_samples": len(valid_indices),
    }


def compute_f1_scores(hpa_labels_list, probs_dict, all_filenames, prefix="true", threshold=0.15, y_true=None):
    if y_true is None:
        y_true = convert_hpa_labels_to_binary(hpa_labels_list, all_filenames, get_subcell_class_names())

    y_pred = []
    valid_indices = []
    for i, filename in enumerate(all_filenames):
        sample_id = f"{prefix}_{filename}"
        if sample_id in probs_dict and hpa_labels_list[i] is not None and len(hpa_labels_list[i]) > 0:
            scores = probs_dict[sample_id]
            if np.any(np.isnan(scores)):
                continue
            y_pred.append((scores >= threshold).astype(int))
            valid_indices.append(i)

    if len(valid_indices) == 0:
        return {"macro_f1": -1.0, "micro_f1": -1.0, "n_valid_samples": 0}

    y_true_valid = y_true[valid_indices]
    y_pred = np.array(y_pred)
    return {
        "macro_f1": f1_score(y_true_valid, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true_valid, y_pred, average="micro", zero_division=0),
        "n_valid_samples": len(valid_indices),
    }


def compute_average_precision_scores(
    hpa_labels_list,
    probs_dict,
    all_filenames,
    prefix="true",
    y_true=None,
    min_examples=0,
):
    if y_true is None:
        y_true = convert_hpa_labels_to_binary(hpa_labels_list, all_filenames, get_subcell_class_names())

    y_scores = []
    valid_indices = []
    for i, filename in enumerate(all_filenames):
        sample_id = f"{prefix}_{filename}"
        if sample_id in probs_dict and hpa_labels_list[i] is not None and len(hpa_labels_list[i]) > 0:
            scores = probs_dict[sample_id]
            if np.any(np.isnan(scores)):
                continue
            y_scores.append(scores)
            valid_indices.append(i)

    if len(valid_indices) == 0:
        return {"macro_ap": -1.0, "micro_ap": -1.0, "n_valid_samples": 0}

    y_true_valid = y_true[valid_indices]
    y_scores = np.array(y_scores)

    if min_examples > 0:
        class_counts = np.sum(y_true_valid, axis=0)
        valid_classes = class_counts >= min_examples
        if not np.any(valid_classes):
            return {"macro_ap": -1.0, "micro_ap": -1.0, "n_valid_samples": len(valid_indices)}
        y_true_valid = y_true_valid[:, valid_classes]
        y_scores = y_scores[:, valid_classes]

    return {
        "macro_ap": average_precision_score(y_true_valid, y_scores, average="macro"),
        "micro_ap": average_precision_score(y_true_valid, y_scores, average="micro"),
        "n_valid_samples": len(valid_indices),
    }


#################################################################################
#                              Main                                              #
#################################################################################

def build_subcell_records(images_dir, filenames, prefix):
    """Build SubCellPortable CSV records from existing images."""
    records = []
    for filename in filenames:
        base_name = filename.replace(".npy", "")
        sample_id = f"{prefix}_{base_name}"
        nuc_path = os.path.join(images_dir, f"{sample_id}_nucleus.png")
        mt_path = os.path.join(images_dir, f"{sample_id}_microtubule.png")
        er_path = os.path.join(images_dir, f"{sample_id}_er.png")
        prot_path = os.path.join(images_dir, f"{sample_id}_protein.png")

        records.append({
            "r_image": str(Path(mt_path).absolute()),
            "y_image": str(Path(er_path).absolute()),
            "b_image": str(Path(nuc_path).absolute()),
            "g_image": str(Path(prot_path).absolute()),
            "output_prefix": sample_id,
        })
    return records


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate protein localization on pre-generated images")

    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing images_true/, images_gen/, and generation_metadata.json")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Output JSON path (default: {images_dir}/subcell_results.json)")
    parser.add_argument("--hpa_csv", type=str,
                        default="/n/home08/skambhampati/ar-microscopy-gen/code/data_preparation/hpa_cell-line-download.csv")  # change to your path
    parser.add_argument("--subcell_dir", type=str,
                        default=os.path.expanduser("~/SubCellPortable"))
    parser.add_argument("--f1_threshold", type=float, default=0.15,
                        help="Probability threshold for F1 score binarization (default: 0.15)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Stage 2: SubCell Localization Evaluation")
    print("=" * 60)

    images_dir = Path(args.images_dir)
    images_dir_true = images_dir / "images_true"
    images_dir_gen = images_dir / "images_gen"
    metadata_path = images_dir / "generation_metadata.json"

    # Load metadata
    if not metadata_path.exists():
        raise FileNotFoundError(f"generation_metadata.json not found in {images_dir}. Run generate_images.py first.")

    with open(metadata_path) as f:
        metadata = json.load(f)

    generation_mode = metadata.get("generation_mode", "protein_only")
    if generation_mode == "all_channels":
        print("ERROR: SubCell evaluation is not applicable in all_channels generation mode.")
        print("In all_channels mode, all channels are generated from scratch (no landmark conditioning),")
        print("so subcellular localization comparison against true images is not meaningful.")
        print("Skipping SubCell evaluation.")
        sys.exit(0)

    filenames = metadata["filenames"]
    print(f"Loaded metadata: {len(filenames)} samples")

    # Setup output directories
    results_dir_true = images_dir / "results_true"
    results_dir_gen = images_dir / "results_gen"
    results_dir_true.mkdir(parents=True, exist_ok=True)
    results_dir_gen.mkdir(parents=True, exist_ok=True)

    # Build SubCellPortable CSV records
    print("\nBuilding SubCellPortable CSV records...")
    records_true = build_subcell_records(str(images_dir_true), filenames, "true")
    records_gen = build_subcell_records(str(images_dir_gen), filenames, "gen")

    csv_true = images_dir / "path_list_true.csv"
    csv_gen = images_dir / "path_list_gen.csv"
    create_subcell_csv(records_true, csv_true)
    create_subcell_csv(records_gen, csv_gen)

    # Run SubCellPortable
    print("\nRunning SubCellPortable on true images...")
    result_csv_true = run_subcell_portable(csv_true.absolute(), results_dir_true.absolute(), args.subcell_dir)
    torch.cuda.empty_cache()

    print("\nRunning SubCellPortable on generated images...")
    result_csv_gen = run_subcell_portable(csv_gen.absolute(), results_dir_gen.absolute(), args.subcell_dir)
    torch.cuda.empty_cache()

    # Parse results
    print("\nParsing SubCellPortable results...")
    subcell_true_dict = parse_subcell_results(result_csv_true)
    subcell_gen_dict = parse_subcell_results(result_csv_gen)
    probs_true = parse_subcell_probabilities(result_csv_true)
    probs_gen = parse_subcell_probabilities(result_csv_gen)

    # Load HPA annotations
    print(f"Loading HPA annotations from {args.hpa_csv}...")
    hpa_df = pd.read_csv(args.hpa_csv)
    hpa_lookup = build_localization_lookup(hpa_df)

    # Extract labels
    hpa_labels = []
    subcell_true = []
    subcell_gen = []
    for filename in filenames:
        hpa_loc = get_localization_from_filename(hpa_lookup, filename + ".npy")
        hpa_labels.append(hpa_loc)
        subcell_true.append(subcell_true_dict.get(f"true_{filename}", []))
        subcell_gen.append(subcell_gen_dict.get(f"gen_{filename}", []))

    # Pre-compute y_true
    class_names = get_subcell_class_names()
    y_true = convert_hpa_labels_to_binary(hpa_labels, filenames, class_names)

    # Compute metrics
    print("\nComputing metrics...")
    metrics = {}

    # Top-k accuracy
    metrics["top_3_hpa_vs_subcell_true"] = compute_top_k_multi_label_accuracy(hpa_labels, subcell_true, k=3)
    metrics["top_3_hpa_vs_subcell_gen"] = compute_top_k_multi_label_accuracy(hpa_labels, subcell_gen, k=3)
    metrics["top_1_hpa_vs_subcell_true"] = compute_top_k_multi_label_accuracy(hpa_labels, subcell_true, k=1)
    metrics["top_1_hpa_vs_subcell_gen"] = compute_top_k_multi_label_accuracy(hpa_labels, subcell_gen, k=1)

    # Spearman correlation
    spearman_correlations = compute_spearman_correlation(probs_true, probs_gen, filenames)
    avg_spearman = np.mean(list(spearman_correlations.values())) if spearman_correlations else -1.0
    metrics["avg_spearman_correlation"] = float(avg_spearman)
    metrics["spearman_correlations_per_sample"] = spearman_correlations

    # MLRAP and Coverage Error
    mlrap_true = compute_multilabel_ranking_metrics(hpa_labels, probs_true, filenames, prefix="true", y_true=y_true)
    metrics["hpa_vs_subcell_true_mlrap"] = mlrap_true["mlrap"]
    metrics["hpa_vs_subcell_true_coverage_error"] = mlrap_true["coverage_error"]

    mlrap_gen = compute_multilabel_ranking_metrics(hpa_labels, probs_gen, filenames, prefix="gen", y_true=y_true)
    metrics["hpa_vs_subcell_gen_mlrap"] = mlrap_gen["mlrap"]
    metrics["hpa_vs_subcell_gen_coverage_error"] = mlrap_gen["coverage_error"]

    # Average Precision
    ap_true = compute_average_precision_scores(hpa_labels, probs_true, filenames, prefix="true", y_true=y_true)
    metrics["hpa_vs_subcell_true_macro_ap"] = ap_true["macro_ap"]
    metrics["hpa_vs_subcell_true_micro_ap"] = ap_true["micro_ap"]

    ap_gen = compute_average_precision_scores(hpa_labels, probs_gen, filenames, prefix="gen", y_true=y_true)
    metrics["hpa_vs_subcell_gen_macro_ap"] = ap_gen["macro_ap"]
    metrics["hpa_vs_subcell_gen_micro_ap"] = ap_gen["micro_ap"]

    # F1 scores
    f1_true = compute_f1_scores(hpa_labels, probs_true, filenames, prefix="true", threshold=args.f1_threshold, y_true=y_true)
    metrics["hpa_vs_subcell_true_macro_f1"] = f1_true["macro_f1"]
    metrics["hpa_vs_subcell_true_micro_f1"] = f1_true["micro_f1"]

    f1_gen = compute_f1_scores(hpa_labels, probs_gen, filenames, prefix="gen", threshold=args.f1_threshold, y_true=y_true)
    metrics["hpa_vs_subcell_gen_macro_f1"] = f1_gen["macro_f1"]
    metrics["hpa_vs_subcell_gen_micro_f1"] = f1_gen["micro_f1"]

    # Assemble results
    results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": metadata.get("configuration", {}),
        "num_samples": len(filenames),
        "metrics": metrics,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("SUBCELL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Top-3 HPA vs SubCell(true): {metrics['top_3_hpa_vs_subcell_true']:.4f}")
    print(f"Top-3 HPA vs SubCell(gen):  {metrics['top_3_hpa_vs_subcell_gen']:.4f}")
    print(f"Top-1 HPA vs SubCell(true): {metrics['top_1_hpa_vs_subcell_true']:.4f}")
    print(f"Top-1 HPA vs SubCell(gen):  {metrics['top_1_hpa_vs_subcell_gen']:.4f}")
    print(f"Spearman correlation:       {metrics['avg_spearman_correlation']:.4f}")
    print(f"MLRAP (true):               {metrics['hpa_vs_subcell_true_mlrap']:.4f}")
    print(f"MLRAP (gen):                {metrics['hpa_vs_subcell_gen_mlrap']:.4f}")
    print(f"Coverage Error (true):      {metrics['hpa_vs_subcell_true_coverage_error']:.4f}")
    print(f"Coverage Error (gen):       {metrics['hpa_vs_subcell_gen_coverage_error']:.4f}")
    print(f"Macro AP (true):            {metrics['hpa_vs_subcell_true_macro_ap']:.4f}")
    print(f"Macro AP (gen):             {metrics['hpa_vs_subcell_gen_macro_ap']:.4f}")
    print(f"Micro AP (true):            {metrics['hpa_vs_subcell_true_micro_ap']:.4f}")
    print(f"Micro AP (gen):             {metrics['hpa_vs_subcell_gen_micro_ap']:.4f}")
    print(f"Macro F1 (true):            {metrics['hpa_vs_subcell_true_macro_f1']:.4f}")
    print(f"Macro F1 (gen):             {metrics['hpa_vs_subcell_gen_macro_f1']:.4f}")
    print("=" * 60)

    # Save results
    output_path = args.output_json or str(images_dir / "subcell_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
