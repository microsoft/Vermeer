#!/usr/bin/env python3
"""
Optimized evaluation script for protein localization classification.
Generates protein channel images and evaluates localization predictions using SubCellPortable.

Optimizations over evaluation_protein_loc.py:
1. In-memory generation + decoding: generate tokens and decode to images in a single loop (no disk I/O)
2. Pre-computed binary labels: y_true computed once and reused across all metric functions

Usage:
python evaluations/ca_esm/evaluation_protein_loc_optimized.py \
    --model_checkpoint /path/to/gpt_checkpoint.pt \
    --val_dir /path/to/hpa-grayscale-ca_code_flip_ten_crop_rotate_esm_embed \
    --num_samples 100 \
    --batch_size 16
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

from tqdm import tqdm
from PIL import Image
import pandas as pd
import ast
import time
from datetime import datetime
from scipy.stats import spearmanr
from sklearn.metrics import label_ranking_average_precision_score, coverage_error, f1_score, average_precision_score

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt_ca import GPT_models
from autoregressive.models.generate_ca import (
    generate_with_prefix,
    decode_tokens_to_images,
    decode_tokens_to_images_batched,
    load_validation_codes
)

# Channel identity constants (index → stain name, matches data pipeline ordering)
CHANNEL_NAMES = ['nucleus', 'microtubule', 'er', 'protein']
# Mapping from stain name to SubCellPortable CSV field
CHANNEL_TO_SUBCELL_FIELD = {
    'nucleus': 'b_image',
    'microtubule': 'r_image',
    'er': 'y_image',
    'protein': 'g_image',
}


def decode_and_save_true_images(codes, filenames, vq_model, args, images_dir, device):
    """Decode true images from codes and save to disk. Returns SubCell records."""
    latent_size = args.image_size // args.downsample_size
    tokens_per_channel = args.block_size_per_channel + 2
    records = []

    for batch_start in tqdm(range(0, len(codes), args.decode_batch_size), desc="Decoding true images"):
        batch_end = min(batch_start + args.decode_batch_size, len(codes))
        batch_codes = codes[batch_start:batch_end]
        batch_filenames = filenames[batch_start:batch_end]

        full_codes = []
        for code in batch_codes:
            if len(code.shape) > 1:
                code = code[0]
            full_codes.append(code)

        true_tokens = torch.from_numpy(np.stack(full_codes)).long().to(device)
        true_images = decode_tokens_to_images_batched(
            true_tokens, vq_model, args.n_channels,
            tokens_per_channel, latent_size, args.codebook_embed_dim
        )

        true_np = true_images.mean(dim=2).cpu().numpy()
        batch_records = save_fov_images_for_subcell(
            true_np, batch_filenames, images_dir, prefix='true'
        )
        records.extend(batch_records)

    return records



#################################################################################
#              Helper Functions (identical to original script)                   #
#################################################################################

def build_localization_lookup(df):
    """Pre-build a dictionary from (uniprot, antibody, cell_line, fov) -> localizations list."""
    lookup = {}
    for _, row in df.iterrows():
        key = (row['uniprot'], row['antibody'], row['cell_line'], int(row['fov']))
        loc_str = row['localizations']
        try:
            localizations = ast.literal_eval(loc_str)
            lookup[key] = localizations if localizations else None
        except:
            lookup[key] = None
    return lookup


def get_localization_from_filename(lookup, filename):
    """O(1) lookup version. `lookup` is the dict from build_localization_lookup."""
    parts = filename.replace('.npy', '').split('_')
    if len(parts) < 5 or parts[-2] != 'fov':
        return None
    uniprot = parts[0]
    antibody = parts[1]
    fov = parts[-1]
    cell_line = '_'.join(parts[2:-2])
    return lookup.get((uniprot, antibody, cell_line, int(fov)))


def load_metadata_for_stratification(metadata_dir, val_split):
    """Load metadata CSV for stratification based on validation split."""
    split_to_csv = {
        'train': 'train_samples.csv',
        'val': 'val_samples.csv',
        'val1': 'val1_samples.csv',
        'val2': 'val2_samples.csv',
        'cell_line_holdouts': 'cell_line_holdouts_samples.csv',
    }

    csv_filename = split_to_csv.get(val_split)
    if csv_filename is None:
        raise ValueError(f"Unknown validation split: {val_split}. Valid options: {list(split_to_csv.keys())}")

    csv_path = os.path.join(metadata_dir, csv_filename)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)

    if 'localizations' in df.columns and df['localizations'].dtype == object:
        df['localizations'] = df['localizations'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
        )

    print(f"Loaded {len(df)} samples from metadata")
    return df


def create_stratification_groups(metadata_df, filenames, stratify_type):
    """Create stratification groups for metrics computation."""
    filename_to_metadata = {}
    for _, row in metadata_df.iterrows():
        base_name = row['base_filename']
        filename_to_metadata[base_name] = row

    groups = {}
    for idx, filename in enumerate(filenames):
        base_name = filename.replace('.npy', '')

        if base_name not in filename_to_metadata:
            print(f"Warning: {base_name} not found in metadata, skipping...")
            continue

        metadata_row = filename_to_metadata[base_name]

        if stratify_type == 'cell_line':
            group_key = metadata_row['cell_line']
        elif stratify_type == 'localization_type':
            group_key = metadata_row['protein_type']
        else:
            raise ValueError(f"Unknown stratify_type: {stratify_type}")

        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(idx)

    return groups


def check_existing_images(images_dir_true, images_dir_gen, filenames, channel_names=None):
    """Check if images already exist for given filenames."""
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    missing_filenames = []

    for filename in filenames:
        base_name = filename.replace('.npy', '')

        true_exists = all(
            os.path.exists(os.path.join(images_dir_true, f"true_{base_name}_{ch}.png"))
            for ch in channel_names
        )
        gen_exists = all(
            os.path.exists(os.path.join(images_dir_gen, f"gen_{base_name}_{ch}.png"))
            for ch in channel_names
        )

        if not (true_exists and gen_exists):
            missing_filenames.append(filename)

    all_exist = len(missing_filenames) == 0
    return all_exist, missing_filenames


def load_models(args, device):
    """Load VQ tokenizer and GPT-CA model."""
    print("Loading VQ tokenizer...")
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim
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
        model_type=args.model_type
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


def save_fov_images_for_subcell(images, filenames, output_dir, prefix='true'):
    """
    Save full FOV images for SubCellPortable processing.

    Stain identity is preserved via channel_config: position i in `images` corresponds
    to stain CHANNEL_NAMES[channel_config[i]], regardless of the order channels appear.

    Args:
        images: (B, n_channels, H, W) numpy array
        filenames: List of filenames
        output_dir: Output directory for FOV images
        prefix: Prefix for saved files (e.g., 'true' or 'gen')
        channel_config: list of original channel indices present in `images` (default: [0,1,2,3])

    Returns:
        List of dicts with paths for each FOV (for SubCellPortable CSV)
    """
    channel_config = list(range(images.shape[1]))

    os.makedirs(output_dir, exist_ok=True)
    records = []

    for i, filename in enumerate(filenames):
        sample_id = f"{prefix}_{filename}"
        stain_paths = {}

        for pos, orig_ch in enumerate(channel_config):
            stain_name = CHANNEL_NAMES[orig_ch]
            ch_img = images[i, pos]
            ch_img = ((ch_img - ch_img.min()) / (ch_img.max() - ch_img.min() + 1e-8) * 255).astype(np.uint8)
            path = os.path.join(output_dir, f"{sample_id}_{stain_name}.png")
            Image.fromarray(ch_img).save(path)
            stain_paths[stain_name] = path

        record = {field: '' for field in ['r_image', 'y_image', 'b_image', 'g_image']}
        for stain_name, path in stain_paths.items():
            field = CHANNEL_TO_SUBCELL_FIELD[stain_name]
            record[field] = str(Path(path).absolute())
        record['output_prefix'] = sample_id
        records.append(record)

    return records


def create_subcell_csv(records, csv_path):
    """Create SubCellPortable CSV file."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['r_image', 'y_image', 'b_image', 'g_image', 'output_prefix'])
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
        print(f"SubCellPortable completed successfully")
    except Exception as e:
        existing_module = sys.modules.get(module_name)
        if existing_module is not None and getattr(existing_module, "__file__", None) == process_py:
            if not hasattr(existing_module, "run_inference"):
                sys.modules.pop(module_name, None)
        print(f"ERROR: SubCellPortable failed with exception: {type(e).__name__}: {e}")
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
        top_3_str = row['top_3_classes_names']

        if pd.notna(top_3_str):
            top_3_labels = [label.strip() for label in str(top_3_str).split(',')]
        else:
            top_3_labels = []

        results[sample_id] = top_3_labels[:top_k]

    return results


def parse_subcell_probabilities(result_csv_path):
    """Parse SubCellPortable prediction probabilities for all labels."""
    df = pd.read_csv(result_csv_path)

    prob_cols = [col for col in df.columns if col.startswith('prob')]

    results = {}
    for _, row in df.iterrows():
        sample_id = row.iloc[0]
        probs = row[prob_cols].values.astype(float)
        results[sample_id] = probs

    return results


def get_subcell_class_names():
    """Return the mapping of class indices to class names for SubCellPortable."""
    return [
        "Nucleoplasm",
        "Nuclear membrane",
        "Nucleoli",
        "Nucleoli fibrillar center",
        "Nuclear speckles",
        "Nuclear bodies",
        "Endoplasmic reticulum",
        "Golgi apparatus",
        "Intermediate filaments",
        "Actin filaments",
        "Microtubules",
        "Mitotic spindle",
        "Centrosome",
        "Plasma membrane",
        "Mitochondria",
        "Aggresome",
        "Cytosol",
        "Vesicles",
        "Peroxisomes",
        "Endosomes",
        "Lysosomes",
        "Lipid droplets",
        "Cytoplasmic bodies",
        "No staining",
        "Cell junctions",
        "Focal adhesion sites",
        "Microtubule organizing center",
        "Microtubule ends",
        "Cytokinetic bridge",
        "Midbody",
        "Midbody ring",
    ]


# from inference.py in SubCellPortable
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


def normalize_label(label):
    """Normalize a localization label for comparison."""
    if label is None:
        return None
    label = str(label).lower()
    label = label.replace('_', ' ')
    label = ' '.join(label.split())
    return label


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


def compute_spearman_correlation(probs_true, probs_gen, all_filenames):
    """Compute Spearman correlation of predicted label rankings between true and generated images."""
    correlations = {}

    for filename in all_filenames:
        sample_id_true = f"true_{filename}"
        sample_id_gen = f"gen_{filename}"

        if sample_id_true in probs_true and sample_id_gen in probs_gen:
            true_probs = probs_true[sample_id_true]
            gen_probs = probs_gen[sample_id_gen]
            corr, _ = spearmanr(true_probs, gen_probs)
            correlations[filename] = float(corr)

    return correlations


def compute_multilabel_ranking_metrics(hpa_labels_list, probs_dict, all_filenames, prefix='true', y_true=None):
    """
    Compute multilabel ranking metrics (MLRAP and Coverage Error).

    Optimization: accepts pre-computed y_true to avoid redundant computation.
    """
    if y_true is None:
        class_names = get_subcell_class_names()
        y_true = convert_hpa_labels_to_binary(hpa_labels_list, all_filenames, class_names)

    y_scores = []
    valid_indices = []

    for i, filename in enumerate(all_filenames):
        sample_id = f"{prefix}_{filename}"
        if sample_id in probs_dict and hpa_labels_list[i] is not None and len(hpa_labels_list[i]) > 0:
            y_scores.append(probs_dict[sample_id])
            valid_indices.append(i)

    if len(valid_indices) == 0:
        return {
            'mlrap': -1.0,
            'coverage_error': -1.0,
            'n_valid_samples': 0
        }

    y_true_valid = y_true[valid_indices]
    y_scores = np.array(y_scores)

    mlrap = label_ranking_average_precision_score(y_true_valid, y_scores)
    cov_error = coverage_error(y_true_valid, y_scores)

    return {
        'mlrap': mlrap,
        'coverage_error': cov_error,
        'n_valid_samples': len(valid_indices)
    }


def compute_f1_scores(hpa_labels_list, probs_dict, all_filenames, prefix='true', threshold=0.5, y_true=None):
    """
    Compute macro and micro F1 scores for multilabel classification.

    Optimization: accepts pre-computed y_true to avoid redundant computation.
    """
    if y_true is None:
        class_names = get_subcell_class_names()
        y_true = convert_hpa_labels_to_binary(hpa_labels_list, all_filenames, class_names)

    y_pred = []
    valid_indices = []

    for i, filename in enumerate(all_filenames):
        sample_id = f"{prefix}_{filename}"
        if sample_id in probs_dict and hpa_labels_list[i] is not None and len(hpa_labels_list[i]) > 0:
            probs = probs_dict[sample_id]
            binary_pred = (probs >= threshold).astype(int)
            y_pred.append(binary_pred)
            valid_indices.append(i)

    if len(valid_indices) == 0:
        return {
            'macro_f1': -1.0,
            'micro_f1': -1.0,
            'n_valid_samples': 0
        }

    y_true_valid = y_true[valid_indices]
    y_pred = np.array(y_pred)

    macro_f1 = f1_score(y_true_valid, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true_valid, y_pred, average='micro', zero_division=0)

    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'n_valid_samples': len(valid_indices)
    }


def compute_average_precision_scores(hpa_labels_list, probs_dict, all_filenames, prefix='true', y_true=None):
    """
    Compute macro and micro average precision scores for multilabel classification.

    Optimization: accepts pre-computed y_true to avoid redundant computation.
    """
    if y_true is None:
        class_names = get_subcell_class_names()
        y_true = convert_hpa_labels_to_binary(hpa_labels_list, all_filenames, class_names)

    y_scores = []
    valid_indices = []

    for i, filename in enumerate(all_filenames):
        sample_id = f"{prefix}_{filename}"
        if sample_id in probs_dict and hpa_labels_list[i] is not None and len(hpa_labels_list[i]) > 0:
            y_scores.append(probs_dict[sample_id])
            valid_indices.append(i)

    if len(valid_indices) == 0:
        return {
            'macro_ap': -1.0,
            'micro_ap': -1.0,
            'n_valid_samples': 0
        }

    y_true_valid = y_true[valid_indices]
    y_scores = np.array(y_scores)

    macro_ap = average_precision_score(y_true_valid, y_scores, average='macro')
    micro_ap = average_precision_score(y_true_valid, y_scores, average='micro')

    return {
        'macro_ap': macro_ap,
        'micro_ap': micro_ap,
        'n_valid_samples': len(valid_indices)
    }


def compute_top_k_multi_label_accuracy(true_labels, pred_labels, k=3):
    n_samples = len(true_labels)
    accuracy = 0
    for i in range(n_samples):
        true_label = true_labels[i]
        pred_label = pred_labels[i]
        if true_label is not None and pred_label is not None:
            normalized_true = set(normalize_label(label) for label in true_label if label is not None)
            normalized_pred = [normalize_label(label) for label in pred_label[:k] if label is not None]
            if any(p in normalized_true for p in normalized_pred):
                accuracy += 1
    return accuracy / n_samples


def compute_metrics_for_group(hpa_labels, subcell_true, subcell_gen, probs_true, probs_gen, all_filenames, group_name, y_true=None):
    """
    Compute all metrics for a specific group of samples.

    Optimization: accepts pre-computed y_true to avoid redundant computation.
    """
    # Pre-compute y_true once for this group if not provided
    if y_true is None:
        class_names = get_subcell_class_names()
        y_true = convert_hpa_labels_to_binary(hpa_labels, all_filenames, class_names)

    metrics = {}

    # Compute top-k accuracy
    top_3_hpa_vs_true_acc = compute_top_k_multi_label_accuracy(hpa_labels, subcell_true, k=3)
    top_3_hpa_vs_gen_acc = compute_top_k_multi_label_accuracy(hpa_labels, subcell_gen, k=3)

    metrics['top_3_hpa_vs_subcell_true'] = top_3_hpa_vs_true_acc
    metrics['top_3_hpa_vs_subcell_gen'] = top_3_hpa_vs_gen_acc

    top_1_hpa_vs_true_acc = compute_top_k_multi_label_accuracy(hpa_labels, subcell_true, k=1)
    top_1_hpa_vs_gen_acc = compute_top_k_multi_label_accuracy(hpa_labels, subcell_gen, k=1)

    metrics['top_1_hpa_vs_subcell_true'] = top_1_hpa_vs_true_acc
    metrics['top_1_hpa_vs_subcell_gen'] = top_1_hpa_vs_gen_acc

    # Compute Spearman correlation of rankings
    spearman_correlations = compute_spearman_correlation(probs_true, probs_gen, all_filenames)
    avg_spearman = np.mean(list(spearman_correlations.values())) if len(spearman_correlations) > 0 else -1.0
    metrics['avg_spearman_correlation'] = float(avg_spearman)
    metrics['spearman_correlations_per_sample'] = spearman_correlations

    # Compute MLRAP and Coverage Error (reuse y_true)
    mlrap_metrics_true = compute_multilabel_ranking_metrics(hpa_labels, probs_true, all_filenames, prefix='true', y_true=y_true)
    metrics['hpa_vs_subcell_true_mlrap'] = mlrap_metrics_true['mlrap']
    metrics['hpa_vs_subcell_true_coverage_error'] = mlrap_metrics_true['coverage_error']

    mlrap_metrics_gen = compute_multilabel_ranking_metrics(hpa_labels, probs_gen, all_filenames, prefix='gen', y_true=y_true)
    metrics['hpa_vs_subcell_gen_mlrap'] = mlrap_metrics_gen['mlrap']
    metrics['hpa_vs_subcell_gen_coverage_error'] = mlrap_metrics_gen['coverage_error']

    # Compute Average Precision scores (reuse y_true)
    ap_metrics_true = compute_average_precision_scores(hpa_labels, probs_true, all_filenames, prefix='true', y_true=y_true)
    metrics['hpa_vs_subcell_true_macro_ap'] = ap_metrics_true['macro_ap']
    metrics['hpa_vs_subcell_true_micro_ap'] = ap_metrics_true['micro_ap']

    ap_metrics_gen = compute_average_precision_scores(hpa_labels, probs_gen, all_filenames, prefix='gen', y_true=y_true)
    metrics['hpa_vs_subcell_gen_macro_ap'] = ap_metrics_gen['macro_ap']
    metrics['hpa_vs_subcell_gen_micro_ap'] = ap_metrics_gen['micro_ap']

    return metrics


def pad_and_batch_esm_embeddings(batch_labels):
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


#################################################################################
#                              Argument Parsing                                 #
#################################################################################

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimized evaluation of protein localization classification using SubCellPortable"
    )

    # Required arguments
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to GPT-CA model checkpoint"
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="Path to validation data directory (contains ca{image_size}_codes/val and ca{image_size}_labels/val)"
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="val",
        help="Validation split (default: val)"
    )

    # Model configuration
    parser.add_argument(
        "--vq_ckpt",
        type=str,
        default="/n/holylabs/chenf2011_lab/Everyone/Sandeep/code/CA_LlamaGen/pretrained_models/vq_ds16_c2i.pt",
        help="Path to VQ tokenizer checkpoint"
    )
    parser.add_argument(
        "--gpt_model",
        type=str,
        default="GPT-B",
        help="GPT model size (default: GPT-B)"
    )
    parser.add_argument(
        "--vq_model",
        type=str,
        default="VQ-16",
        help="VQ model type (default: VQ-16)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ca_esm_embed_mean_pool",
        help="Model type (default: ca_esm_embed_mean_pool)"
    )
    parser.add_argument(
        "--codebook_size",
        type=int,
        default=16384,
        help="Codebook size (default: 16384)"
    )
    parser.add_argument(
        "--codebook_embed_dim",
        type=int,
        default=8,
        help="Codebook embedding dimension (default: 8)"
    )
    parser.add_argument(
        "--n_channels",
        type=int,
        default=4,
        help="Number of channels (default: 4)"
    )
    parser.add_argument(
        "--n_max_channels",
        type=int,
        default=5,
        help="Maximum channels model supports (default: 5)"
    )
    parser.add_argument(
        "--channel_config",
        type=str,
        default=None,
        help="Comma-separated channel indices to select and order, e.g. '0,1,3'. Default: use all channels in original order."
    )
    parser.add_argument(
        "--block_size_per_channel",
        type=int,
        default=256,
        help="Patches per channel (default: 256)"
    )
    parser.add_argument(
        "--n_prefix_channels",
        type=int,
        default=3,
        help="Number of prefix channels (default: 3)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size (default: 256)"
    )
    parser.add_argument(
        "--downsample_size",
        type=int,
        default=16,
        help="Downsample size (default: 16)"
    )

    # Evaluation configuration
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for AR generation (default: 4)"
    )
    parser.add_argument(
        "--decode_batch_size",
        type=int,
        default=64,
        help="Batch size for VQ decoding of true images (default: 64)"
    )
    # SubCellPortable configuration
    parser.add_argument(
        "--subcell_dir",
        type=str,
        default=os.path.expanduser("~/SubCellPortable"),
        help="Path to SubCellPortable directory"
    )
    parser.add_argument(
        "--subcell_output",
        type=str,
        default="./subcell_eval_output",
        help="Output directory for SubCellPortable"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Optional prefix for output directory (e.g., 'exp001'). If provided, will be appended to subcell_output"
    )

    # HPA annotation
    parser.add_argument(
        "--hpa_csv",
        type=str,
        default="/n/home08/skambhampati/ar-microscopy-gen/code/data_preparation/hpa_cell-line-download.csv",
        help="Path to HPA annotation CSV"
    )

    # Stratification parameters
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default=None,
        help="Path to metadata directory (required if --stratify is used)"
    )
    parser.add_argument(
        "--stratify",
        type=str,
        default=None,
        choices=['cell_line', 'localization_type', None],
        help="Stratification type: 'cell_line' or 'localization_type' (default: None)"
    )

    # Sampling parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2000,
        help="Top-k sampling (default: 2000)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling (default: 1.0)"
    )

    # Output configuration
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON file for results (default: protein_loc_results.json in output directory)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--true_subcell_results",
        type=str,
        default=None,
        help="Path to precomputed true_precomputed.json (skips true image decoding and SubCellPortable run)"
    )

    parser.add_argument(
        "--no-compile",
        action='store_true',
        help="Disable torch.compile for debugging"
    )

    return parser.parse_args()


#################################################################################
#                   Precompute True Image SubCellPortable Results                #
#################################################################################

def preprocess_true_images(val_dir, val_split, image_size, vq_ckpt, vq_model_name,
                           codebook_size, codebook_embed_dim, n_channels,
                           block_size_per_channel, downsample_size,
                           output_dir, subcell_dir, hpa_csv,
                           num_samples=None, decode_batch_size=64, seed=42):
    """
    Precompute true image decoding and SubCellPortable results once.

    This avoids redundant work when evaluating multiple checkpoints, since
    true images only depend on validation data + VQ decoder, not the GPT checkpoint.

    Args:
        val_dir: Path to validation data directory
        val_split: Validation split name
        image_size: Image size
        vq_ckpt: Path to VQ tokenizer checkpoint
        vq_model_name: VQ model type (e.g., "VQ-16")
        codebook_size: Codebook size
        codebook_embed_dim: Codebook embedding dimension
        n_channels: Number of channels
        block_size_per_channel: Patches per channel
        downsample_size: Downsample size
        output_dir: Output directory for precomputed results
        subcell_dir: Path to SubCellPortable directory
        hpa_csv: Path to HPA annotation CSV
        num_samples: Number of samples (None = all)
        decode_batch_size: Batch size for VQ decoding
        seed: Random seed

    Returns:
        str: Path to true_precomputed.json
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    precomputed_json = output_dir / "true_precomputed.json"

    # Check if already computed
    if precomputed_json.exists():
        print(f"Precomputed true results already exist: {precomputed_json}")
        return str(precomputed_json)

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir_true = output_dir / "images_true"
    results_dir_true = output_dir / "results_true"
    images_dir_true.mkdir(parents=True, exist_ok=True)
    results_dir_true.mkdir(parents=True, exist_ok=True)

    # Load VQ model
    print("Loading VQ tokenizer for true image preprocessing...")
    vq_model = VQ_models[vq_model_name](
        codebook_size=codebook_size,
        codebook_embed_dim=codebook_embed_dim
    )
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"VQ tokenizer loaded from {vq_ckpt}")

    # Load validation codes
    code_dir = os.path.join(val_dir, f"ca{image_size}_codes", val_split)
    label_dir = os.path.join(val_dir, f"ca{image_size}_labels", val_split)

    print(f"Loading validation data from {val_dir}...")
    codes, labels, filenames = load_validation_codes(code_dir, label_dir, aug_idx=0)

    if num_samples is not None:
        codes = codes[:num_samples]
        labels = labels[:num_samples]
        filenames = filenames[:num_samples]

    print(f"Processing {len(filenames)} true samples...")

    # Check if true images already exist
    all_exist, missing = check_existing_images(images_dir_true, images_dir_true, filenames)
    # Re-check only true images (the check_existing_images also checks gen, so do manual check)
    true_all_exist = True
    for filename in filenames:
        base_name = filename.replace('.npy', '')
        for ch in ['nucleus', 'microtubule', 'er', 'protein']:
            if not os.path.exists(os.path.join(images_dir_true, f"true_{base_name}_{ch}.png")):
                true_all_exist = False
                break
        if not true_all_exist:
            break

    latent_size = image_size // downsample_size
    tokens_per_channel = block_size_per_channel + 2
    records_true = []

    if true_all_exist:
        print("All true images already exist. Skipping decoding...")
        for filename in filenames:
            base_name = filename.replace('.npy', '')
            fov_nuc_path = os.path.join(images_dir_true, f"true_{base_name}_nucleus.png")
            fov_mt_path = os.path.join(images_dir_true, f"true_{base_name}_microtubule.png")
            fov_er_path = os.path.join(images_dir_true, f"true_{base_name}_er.png")
            fov_prot_path = os.path.join(images_dir_true, f"true_{base_name}_protein.png")
            records_true.append({
                'r_image': str(Path(fov_mt_path).absolute()),
                'y_image': str(Path(fov_er_path).absolute()),
                'b_image': str(Path(fov_nuc_path).absolute()),
                'g_image': str(Path(fov_prot_path).absolute()),
                'output_prefix': f"true_{base_name}"
            })
    else:
        print(f"Decoding true tokens to images (decode_batch_size={decode_batch_size})...")

        # Build a simple namespace with the args that decode_and_save_true_images expects
        class _Args:
            pass
        _args = _Args()
        _args.image_size = image_size
        _args.downsample_size = downsample_size
        _args.block_size_per_channel = block_size_per_channel
        _args.n_channels = n_channels
        _args.codebook_embed_dim = codebook_embed_dim
        _args.decode_batch_size = decode_batch_size

        records_true = decode_and_save_true_images(
            codes, filenames, vq_model, _args, images_dir_true, device
        )

        print(f"Decoded and saved {len(filenames)} true images")

    # Free VQ model
    del vq_model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Create SubCellPortable CSV
    csv_true = output_dir / "path_list_true.csv"
    create_subcell_csv(records_true, csv_true)
    print(f"Created SubCellPortable CSV: {csv_true}")

    # Run SubCellPortable on true images
    print("Running SubCellPortable on true images...")
    result_csv_true = run_subcell_portable(csv_true.absolute(), results_dir_true.absolute(), subcell_dir)
    torch.cuda.empty_cache()
    print(f"SubCellPortable results saved to: {result_csv_true}")

    # Save precomputed metadata
    precomputed_data = {
        'filenames': filenames,
        'result_csv_true': str(result_csv_true),
        'images_dir_true': str(images_dir_true),
        'num_samples': len(filenames),
        'val_dir': val_dir,
        'val_split': val_split,
        'image_size': image_size,
    }
    with open(precomputed_json, 'w') as f:
        json.dump(precomputed_data, f, indent=2)
    print(f"Precomputed true results saved to: {precomputed_json}")

    return str(precomputed_json)


#################################################################################
#                           Main Evaluation Function                            #
#################################################################################

def evaluate(args):
    """Main evaluation function with decoupled generation and decoding."""
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Validate stratification parameters
    if args.stratify and not args.metadata_dir:
        raise ValueError("--metadata_dir is required when --stratify is used")

    # Create output directories with optional prefix
    if args.output_prefix:
        subcell_output = Path(args.subcell_output) / args.output_prefix
        print(f"Using output prefix: {args.output_prefix}")
    else:
        subcell_output = Path(args.subcell_output)

    images_dir_true = subcell_output / "images_true"
    images_dir_gen = subcell_output / "images_gen"
    results_dir_true = subcell_output / "results_true"
    results_dir_gen = subcell_output / "results_gen"
    for d in [images_dir_true, images_dir_gen, results_dir_true, results_dir_gen]:
        d.mkdir(parents=True, exist_ok=True)

    # Load models
    vq_model, gpt_model = load_models(args, device)

    if not args.no_compile:
        print("Compiling GPT model with torch.compile...")
        gpt_model = torch.compile(gpt_model)

    # Load HPA annotations
    print(f"Loading HPA annotations from {args.hpa_csv}...")
    hpa_df = pd.read_csv(args.hpa_csv)
    print(f"Loaded {len(hpa_df)} HPA annotations")
    hpa_lookup = build_localization_lookup(hpa_df)

    # Load validation data
    code_dir = os.path.join(args.val_dir, f"ca{args.image_size}_codes", args.val_split)
    label_dir = os.path.join(args.val_dir, f"ca{args.image_size}_labels", args.val_split)

    print(f"Loading validation data from {args.val_dir}...")
    codes, labels, filenames = load_validation_codes(code_dir, label_dir, aug_idx=0)

    if args.num_samples is not None:
        codes = codes[:args.num_samples]
        labels = labels[:args.num_samples]
        filenames = filenames[:args.num_samples]

    print(f"Processing {len(filenames)} samples...")

    # Apply channel_config: select and reorder channel blocks in each code
    if args.channel_config is not None:
        channel_config = [int(x) for x in args.channel_config.split(',')]
        tpc = args.block_size_per_channel + 2  # tokens per channel (SOC + codes + EOC)
        new_codes = []
        for code in codes:
            if len(code.shape) > 1:
                # Shape (num_aug, seq_len)
                eos = code[:, -1:]
                blocks = [code[:, i * tpc:(i + 1) * tpc] for i in range(args.n_channels)]
                selected = np.concatenate([blocks[ch] for ch in channel_config] + [eos], axis=1)
            else:
                # Shape (seq_len,)
                eos = code[-1:]
                blocks = [code[i * tpc:(i + 1) * tpc] for i in range(args.n_channels)]
                selected = np.concatenate([blocks[ch] for ch in channel_config] + [eos])
            new_codes.append(selected)
        codes = new_codes
        args.n_channels = len(channel_config)
        print(f"channel_config={channel_config} → using {args.n_channels} channels")

        # Stain names present after channel selection (used for image existence checks and saving)
        active_channel_names = [CHANNEL_NAMES[ch] for ch in channel_config]
    else:
        active_channel_names = CHANNEL_NAMES

    # Load metadata if stratification is requested
    stratification_groups = None
    metadata_df = None
    if args.stratify:
        metadata_df = load_metadata_for_stratification(args.metadata_dir, args.val_split)
        stratification_groups = create_stratification_groups(metadata_df, filenames, args.stratify)
        print(f"\nStratification by {args.stratify}:")
        for group_name, indices in sorted(stratification_groups.items()):
            print(f"  {group_name}: {len(indices)} samples")

    # Batch processing setup
    latent_size = args.image_size // args.downsample_size
    tokens_per_channel = args.block_size_per_channel + 2
    all_filenames = []
    records_true = []
    records_gen = []

    # Determine if we're using precomputed true results
    use_precomputed_true = args.true_subcell_results is not None

    # Check if images already exist
    if use_precomputed_true:
        # Only check generated images when using precomputed true results
        gen_all_exist = True
        for filename in filenames:
            base_name = filename.replace('.npy', '')
            for ch in active_channel_names:
                if not os.path.exists(os.path.join(images_dir_gen, f"gen_{base_name}_{ch}.png")):
                    gen_all_exist = False
                    break
            if not gen_all_exist:
                break
        all_exist = gen_all_exist
        missing_filenames = [] if all_exist else filenames
    else:
        all_exist, missing_filenames = check_existing_images(images_dir_true, images_dir_gen, filenames, channel_names=active_channel_names)

    if all_exist:
        print("\nAll images already exist. Skipping generation...")
        skip_generation = True
        all_filenames = filenames
    elif len(missing_filenames) > 0:
        print(f"\n{len(missing_filenames)} images missing. Will generate them.")
        skip_generation = False
    else:
        skip_generation = False

    if not skip_generation:
        # =====================================================================
        # In-memory generation + decoding loop
        # =====================================================================
        print(f"\n--- Generating and decoding samples (batch_size={args.batch_size}) ---")
        for batch_start in tqdm(range(0, len(codes), args.batch_size), desc="Generating samples"):
            batch_end = min(batch_start + args.batch_size, len(codes))
            batch_codes = codes[batch_start:batch_end]
            batch_labels = labels[batch_start:batch_end]
            batch_filenames = filenames[batch_start:batch_end]

            # Extract prefix tokens (first n_prefix_channels)
            prefix_len = args.n_prefix_channels * tokens_per_channel
            prefix_tokens = []
            full_codes = []

            for code in batch_codes:
                if len(code.shape) > 1:
                    code = code[0]
                prefix_tokens.append(code[:prefix_len])
                full_codes.append(code)

            prefix_tokens = torch.from_numpy(np.stack(prefix_tokens)).long().to(device)

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
            cond = cond.to(dtype=next(gpt_model.parameters()).dtype)

            # Generate with mixed precision
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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
                    sample_logits=True
                )

            # Remove EOS token
            generated_tokens = generated_tokens[:, :-1]

            # Decode generated images in memory
            generated_images = decode_tokens_to_images(
                generated_tokens, vq_model, args.n_channels,
                tokens_per_channel, latent_size, args.codebook_embed_dim
            )
            gen_np = generated_images.mean(dim=2).cpu().numpy()
            gen_records = save_fov_images_for_subcell(gen_np, batch_filenames, images_dir_gen, prefix='gen')
            records_gen.extend(gen_records)

            # Decode true images in memory (if not using precomputed)
            if not use_precomputed_true:
                true_tokens = torch.from_numpy(np.stack(full_codes)).long().to(device)
                true_images = decode_tokens_to_images(
                    true_tokens, vq_model, args.n_channels,
                    tokens_per_channel, latent_size, args.codebook_embed_dim
                )
                true_np = true_images.mean(dim=2).cpu().numpy()
                true_records = save_fov_images_for_subcell(true_np, batch_filenames, images_dir_true, prefix='true')
                records_true.extend(true_records)

            all_filenames.extend(batch_filenames)

        print(f"\nGeneration complete: processed {len(all_filenames)} samples")

    else:
        print("Loading existing image filenames...")
        # Build records from existing images
        for filename in all_filenames:
            base_name = filename.replace('.npy', '')

            if not use_precomputed_true:
                fov_nuc_path = os.path.join(images_dir_true, f"true_{base_name}_nucleus.png")
                fov_mt_path = os.path.join(images_dir_true, f"true_{base_name}_microtubule.png")
                fov_er_path = os.path.join(images_dir_true, f"true_{base_name}_er.png")
                fov_prot_path = os.path.join(images_dir_true, f"true_{base_name}_protein.png")

                records_true.append({
                    'r_image': str(Path(fov_mt_path).absolute()),
                    'y_image': str(Path(fov_er_path).absolute()),
                    'b_image': str(Path(fov_nuc_path).absolute()),
                    'g_image': str(Path(fov_prot_path).absolute()),
                    'output_prefix': f"true_{base_name}"
                })

            fov_nuc_path = os.path.join(images_dir_gen, f"gen_{base_name}_nucleus.png")
            fov_mt_path = os.path.join(images_dir_gen, f"gen_{base_name}_microtubule.png")
            fov_er_path = os.path.join(images_dir_gen, f"gen_{base_name}_er.png")
            fov_prot_path = os.path.join(images_dir_gen, f"gen_{base_name}_protein.png")

            records_gen.append({
                'r_image': str(Path(fov_mt_path).absolute()),
                'y_image': str(Path(fov_er_path).absolute()),
                'b_image': str(Path(fov_nuc_path).absolute()),
                'g_image': str(Path(fov_prot_path).absolute()),
                'output_prefix': f"gen_{base_name}"
            })

    if use_precomputed_true:
        print(f"\nUsing precomputed true SubCellPortable results from: {args.true_subcell_results}")
        with open(args.true_subcell_results, 'r') as f:
            precomputed = json.load(f)
        result_csv_true = precomputed['result_csv_true']
        print(f"  True result CSV: {result_csv_true}")

        # Only create CSV and run SubCellPortable for generated images
        print("\nCreating SubCellPortable CSV for generated images...")
        csv_gen = subcell_output / "path_list_gen.csv"
        create_subcell_csv(records_gen, csv_gen)
        print(f"  Generated images CSV: {csv_gen.absolute()}")

        torch.cuda.empty_cache()

        print("\nRunning SubCellPortable on generated images...")
        result_csv_gen = run_subcell_portable(csv_gen.absolute(), results_dir_gen.absolute(), args.subcell_dir)
        torch.cuda.empty_cache()
    else:
        # Create CSVs for SubCellPortable
        print("\nCreating SubCellPortable CSV files...")
        csv_true = subcell_output / "path_list_true.csv"
        csv_gen = subcell_output / "path_list_gen.csv"

        create_subcell_csv(records_true, csv_true)
        create_subcell_csv(records_gen, csv_gen)

        print(f"Created CSV files:")
        print(f"  True images: {csv_true.absolute()}")
        print(f"  Generated images: {csv_gen.absolute()}")

        torch.cuda.empty_cache()

        # Run SubCellPortable (two separate runs: true, gen)
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

    print("Parsing prediction probabilities for all labels...")
    probs_true = parse_subcell_probabilities(result_csv_true)
    probs_gen = parse_subcell_probabilities(result_csv_gen)

    # Extract HPA labels
    hpa_labels = []
    subcell_true = []
    subcell_gen = []

    for filename in all_filenames:
        hpa_loc = get_localization_from_filename(hpa_lookup, filename + '.npy')
        hpa_labels.append(hpa_loc)
        subcell_true.append(subcell_true_dict.get(f"true_{filename}", []))
        subcell_gen.append(subcell_gen_dict.get(f"gen_{filename}", []))

    # Debug: Print first few samples to check label formats
    print("\n" + "="*60)
    print("DEBUG: Checking label formats (first 3 samples):")
    print("="*60)
    for i in range(min(3, len(all_filenames))):
        print(f"\nSample {i+1}: {all_filenames[i]}")
        print(f"  HPA labels (raw): {hpa_labels[i]}")
        if hpa_labels[i]:
            print(f"  HPA labels (normalized): {[normalize_label(l) for l in hpa_labels[i]]}")
        print(f"  SubCell (true): {subcell_true[i]}")
        if subcell_true[i]:
            print(f"  SubCell (true, normalized): {[normalize_label(l) for l in subcell_true[i]]}")
        print(f"  SubCell (gen): {subcell_gen[i]}")
        if subcell_gen[i]:
            print(f"  SubCell (gen, normalized): {[normalize_label(l) for l in subcell_gen[i]]}")

    # Clear GPU memory
    print("\nClearing GPU memory...")
    if 'vq_model' in dir():
        del vq_model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print(f"GPU memory freed")

    print("="*60 + "\n")

    # Pre-compute y_true once for all metric functions (Optimization #3)
    class_names = get_subcell_class_names()
    y_true_all = convert_hpa_labels_to_binary(hpa_labels, all_filenames, class_names)

    # Compute metrics (stratified or overall)
    if args.stratify:
        print(f"\n{'='*60}")
        print(f"COMPUTING STRATIFIED METRICS (by {args.stratify})")
        print(f"{'='*60}")

        metrics = {}
        metrics['stratification_type'] = args.stratify
        metrics['stratified_results'] = {}

        for group_name, group_indices in sorted(stratification_groups.items()):
            print(f"\nComputing metrics for {args.stratify}={group_name}...")

            group_filenames = [all_filenames[i] for i in group_indices]
            group_hpa_labels = [hpa_labels[i] for i in group_indices]
            group_subcell_true = [subcell_true[i] for i in group_indices]
            group_subcell_gen = [subcell_gen[i] for i in group_indices]

            # Pre-compute y_true for this group
            group_y_true = convert_hpa_labels_to_binary(group_hpa_labels, group_filenames, class_names)

            group_metrics = compute_metrics_for_group(
                group_hpa_labels, group_subcell_true, group_subcell_gen,
                probs_true, probs_gen, group_filenames, group_name,
                y_true=group_y_true
            )

            metrics['stratified_results'][group_name] = group_metrics
            metrics['stratified_results'][group_name]['n_samples'] = len(group_filenames)

        # Compute overall metrics
        print(f"\nComputing overall (non-stratified) metrics...")
        overall_metrics = compute_metrics_for_group(
            hpa_labels, subcell_true, subcell_gen,
            probs_true, probs_gen, all_filenames, 'overall',
            y_true=y_true_all
        )
        metrics['overall'] = overall_metrics
        metrics['overall']['n_samples'] = len(all_filenames)

    else:
        metrics = {}

        top_3_hpa_vs_true_acc = compute_top_k_multi_label_accuracy(hpa_labels, subcell_true, k=3)
        top_3_hpa_vs_gen_acc = compute_top_k_multi_label_accuracy(hpa_labels, subcell_gen, k=3)

        metrics['top_3_hpa_vs_subcell_true'] = top_3_hpa_vs_true_acc
        metrics['top_3_hpa_vs_subcell_gen'] = top_3_hpa_vs_gen_acc

        top_1_hpa_vs_true_acc = compute_top_k_multi_label_accuracy(hpa_labels, subcell_true, k=1)
        top_1_hpa_vs_gen_acc = compute_top_k_multi_label_accuracy(hpa_labels, subcell_gen, k=1)

        metrics['top_1_hpa_vs_subcell_true'] = top_1_hpa_vs_true_acc
        metrics['top_1_hpa_vs_subcell_gen'] = top_1_hpa_vs_gen_acc

        # Compute Spearman correlation of rankings
        print("Computing Spearman correlation of label rankings...")
        spearman_correlations = compute_spearman_correlation(probs_true, probs_gen, all_filenames)
        avg_spearman = np.mean(list(spearman_correlations.values())) if len(spearman_correlations) > 0 else -1.0
        metrics['avg_spearman_correlation'] = float(avg_spearman)
        metrics['spearman_correlations_per_sample'] = spearman_correlations

        # Compute MLRAP and Coverage Error (reuse y_true_all)
        print("Computing MLRAP and Coverage Error for HPA vs SubCell(true)...")
        mlrap_metrics_true = compute_multilabel_ranking_metrics(hpa_labels, probs_true, all_filenames, prefix='true', y_true=y_true_all)
        metrics['hpa_vs_subcell_true_mlrap'] = mlrap_metrics_true['mlrap']
        metrics['hpa_vs_subcell_true_coverage_error'] = mlrap_metrics_true['coverage_error']

        print("Computing MLRAP and Coverage Error for HPA vs SubCell(gen)...")
        mlrap_metrics_gen = compute_multilabel_ranking_metrics(hpa_labels, probs_gen, all_filenames, prefix='gen', y_true=y_true_all)
        metrics['hpa_vs_subcell_gen_mlrap'] = mlrap_metrics_gen['mlrap']
        metrics['hpa_vs_subcell_gen_coverage_error'] = mlrap_metrics_gen['coverage_error']

        # Compute Average Precision scores (reuse y_true_all)
        print("Computing Average Precision scores for HPA vs SubCell(true)...")
        ap_metrics_true = compute_average_precision_scores(hpa_labels, probs_true, all_filenames, prefix='true', y_true=y_true_all)
        metrics['hpa_vs_subcell_true_macro_ap'] = ap_metrics_true['macro_ap']
        metrics['hpa_vs_subcell_true_micro_ap'] = ap_metrics_true['micro_ap']

        print("Computing Average Precision scores for HPA vs SubCell(gen)...")
        ap_metrics_gen = compute_average_precision_scores(hpa_labels, probs_gen, all_filenames, prefix='gen', y_true=y_true_all)
        metrics['hpa_vs_subcell_gen_macro_ap'] = ap_metrics_gen['macro_ap']
        metrics['hpa_vs_subcell_gen_micro_ap'] = ap_metrics_gen['micro_ap']

    # Prepare results
    results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'model_checkpoint': args.model_checkpoint,
            'val_dir': args.val_dir,
            'num_samples': len(all_filenames),
            'batch_size': args.batch_size,
            'n_prefix_channels': args.n_prefix_channels,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'top_p': args.top_p,
            'stratify': args.stratify,
            'metadata_dir': args.metadata_dir if args.stratify else None
        },
        'metrics': metrics,
        'per_sample_details': []
    }

    # Print results
    if args.stratify:
        print("\n" + "="*60)
        print(f"STRATIFIED EVALUATION RESULTS (by {args.stratify})")
        print("="*60)

        for group_name, group_metrics in sorted(metrics['stratified_results'].items()):
            print(f"\n{'-'*60}")
            print(f"GROUP: {group_name} (n={group_metrics['n_samples']})")
            print(f"{'-'*60}")
            print(f"Top-3 HPA vs SubCell(true): {group_metrics['top_3_hpa_vs_subcell_true']:.4f}")
            print(f"Top-3 HPA vs SubCell(gen):  {group_metrics['top_3_hpa_vs_subcell_gen']:.4f}")
            print(f"Top-1 HPA vs SubCell(true): {group_metrics['top_1_hpa_vs_subcell_true']:.4f}")
            print(f"Top-1 HPA vs SubCell(gen):  {group_metrics['top_1_hpa_vs_subcell_gen']:.4f}")
            print(f"Spearman correlation:       {group_metrics['avg_spearman_correlation']:.4f}")
            print(f"MLRAP (true):               {group_metrics['hpa_vs_subcell_true_mlrap']:.4f}")
            print(f"MLRAP (gen):                {group_metrics['hpa_vs_subcell_gen_mlrap']:.4f}")
            print(f"Coverage Error (true):      {group_metrics['hpa_vs_subcell_true_coverage_error']:.4f}")
            print(f"Coverage Error (gen):       {group_metrics['hpa_vs_subcell_gen_coverage_error']:.4f}")
            print(f"Macro AP (true):            {group_metrics['hpa_vs_subcell_true_macro_ap']:.4f}")
            print(f"Macro AP (gen):             {group_metrics['hpa_vs_subcell_gen_macro_ap']:.4f}")
            print(f"Micro AP (true):            {group_metrics['hpa_vs_subcell_true_micro_ap']:.4f}")
            print(f"Micro AP (gen):             {group_metrics['hpa_vs_subcell_gen_micro_ap']:.4f}")

        print("\n" + "="*60)
        print("OVERALL METRICS (all samples)")
        print("="*60)
        overall_metrics = metrics['overall']
        print(f"Total samples: {overall_metrics['n_samples']}")
        print(f"\nTop-3 HPA vs SubCell(true): {overall_metrics['top_3_hpa_vs_subcell_true']:.4f}")
        print(f"Top-3 HPA vs SubCell(gen):  {overall_metrics['top_3_hpa_vs_subcell_gen']:.4f}")
        print(f"Top-1 HPA vs SubCell(true): {overall_metrics['top_1_hpa_vs_subcell_true']:.4f}")
        print(f"Top-1 HPA vs SubCell(gen):  {overall_metrics['top_1_hpa_vs_subcell_gen']:.4f}")
        print(f"Spearman correlation:       {overall_metrics['avg_spearman_correlation']:.4f}")
        print(f"MLRAP (true):               {overall_metrics['hpa_vs_subcell_true_mlrap']:.4f}")
        print(f"MLRAP (gen):                {overall_metrics['hpa_vs_subcell_gen_mlrap']:.4f}")
        print(f"Coverage Error (true):      {overall_metrics['hpa_vs_subcell_true_coverage_error']:.4f}")
        print(f"Coverage Error (gen):       {overall_metrics['hpa_vs_subcell_gen_coverage_error']:.4f}")
        print(f"Macro AP (true):            {overall_metrics['hpa_vs_subcell_true_macro_ap']:.4f}")
        print(f"Macro AP (gen):             {overall_metrics['hpa_vs_subcell_gen_macro_ap']:.4f}")
        print(f"Micro AP (true):            {overall_metrics['hpa_vs_subcell_true_micro_ap']:.4f}")
        print(f"Micro AP (gen):             {overall_metrics['hpa_vs_subcell_gen_micro_ap']:.4f}")
        print("\n" + "="*60)
    else:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)

        print("\n" + "-"*60)
        print("TOP-K ACCURACY METRICS")
        print("-"*60)
        print(f"HPA vs SubCell(true) top-3:          {metrics['top_3_hpa_vs_subcell_true']:.4f}")
        print(f"HPA vs SubCell(gen) top-3:           {metrics['top_3_hpa_vs_subcell_gen']:.4f}")
        print(f"HPA vs SubCell(true) top-1:          {metrics['top_1_hpa_vs_subcell_true']:.4f}")
        print(f"HPA vs SubCell(gen) top-1:           {metrics['top_1_hpa_vs_subcell_gen']:.4f}")

        print("\n" + "-"*60)
        print("SPEARMAN CORRELATION")
        print("-"*60)
        print(f"Average correlation (True vs Gen rankings): {metrics['avg_spearman_correlation']:.4f}")

        print("\n" + "-"*60)
        print("MULTILABEL RANKING METRICS")
        print("-"*60)
        print(f"HPA vs SubCell(true):")
        print(f"  - MLRAP:           {metrics['hpa_vs_subcell_true_mlrap']:.4f}")
        print(f"  - Coverage Error:  {metrics['hpa_vs_subcell_true_coverage_error']:.4f}")
        print(f"\nHPA vs SubCell(gen):")
        print(f"  - MLRAP:           {metrics['hpa_vs_subcell_gen_mlrap']:.4f}")
        print(f"  - Coverage Error:  {metrics['hpa_vs_subcell_gen_coverage_error']:.4f}")

        print("\n" + "-"*60)
        print("AVERAGE PRECISION SCORES")
        print("-"*60)
        print(f"HPA vs SubCell(true):")
        print(f"  - Macro AP:        {metrics['hpa_vs_subcell_true_macro_ap']:.4f}")
        print(f"  - Micro AP:        {metrics['hpa_vs_subcell_true_micro_ap']:.4f}")
        print(f"\nHPA vs SubCell(gen):")
        print(f"  - Macro AP:        {metrics['hpa_vs_subcell_gen_macro_ap']:.4f}")
        print(f"  - Micro AP:        {metrics['hpa_vs_subcell_gen_micro_ap']:.4f}")

        print("\n" + "="*60)

    # Save results
    if args.output_json:
        output_path = args.output_json
    else:
        output_path = subcell_output / "protein_loc_results.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    """Main entry point."""
    args = parse_args()

    print("="*60)
    print("Protein Localization Evaluation (Optimized)")
    print("="*60)
    print(f"Model checkpoint: {args.model_checkpoint}")
    print(f"Validation directory: {args.val_dir}")
    if args.output_prefix:
        print(f"Output directory: {args.subcell_output}/{args.output_prefix}")
    else:
        print(f"Output directory: {args.subcell_output}")
    print(f"Generation batch size: {args.batch_size}")
    print(f"Decode batch size: {args.decode_batch_size}")
    print("="*60 + "\n")

    evaluate(args)


if __name__ == "__main__":
    main()
