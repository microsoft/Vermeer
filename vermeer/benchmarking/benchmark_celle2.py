#!/usr/bin/env python3
# Code derived from original celle2 repo (https://github.com/BoHuangLab/CELL-E_2)
"""
Benchmark CELL-E2 (HPA_480) on the same protein localization evaluation used for CA_LlamaGen.

Generates protein channel images conditioned on nucleus images + protein sequences,
then runs SubCellPortable to classify protein localization, comparing predictions
against HPA ground truth labels.

Preprocessing pipeline (matches test_celle2.ipynb):
    1. Load raw TIFF channels from HPA download directory
    2. Percentile clip (0.001/99.999) + rescale to [0, 1] per channel
    3. Center crop to 1024x1024 (one crop per sample)
    4. Resize to 256x256
    5. Apply CELL-E2 z-score normalization on nucleus+protein
    6. Final CELL-E2 scaling on nucleus: divide by abs(max)

Usage:
    python benchmark_celle2.py \
        --val_data_dir /path/to/concatenated_arrays \
        --tiff_root /path/to/raw_tiffs \
        --val_split val1 \
        --output_dir /path/to/output \
        --batch_size 1
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
import pickle
import re

from tqdm import tqdm
from PIL import Image
from tifffile import imread
from torchvision import transforms
import pandas as pd
import ast
import time
from datetime import datetime
from scipy.stats import spearmanr
from sklearn.metrics import (
    label_ranking_average_precision_score,
    coverage_error,
    f1_score,
    average_precision_score,
)

# ---------------------------------------------------------------------------
# Import SubCellPortable evaluation helpers from CA_LlamaGen
# ---------------------------------------------------------------------------
CA_LLAMAGEN_DIR = os.path.join(os.path.dirname(__file__), "..", "CA_LlamaGen")
sys.path.insert(0, os.path.abspath(CA_LLAMAGEN_DIR))
from evaluations.ca_esm.evaluation_protein_loc import (
    save_fov_images_for_subcell,
    create_subcell_csv,
    run_subcell_portable,
    parse_subcell_results,
    parse_subcell_probabilities,
    compute_metrics_for_group,
    build_localization_lookup,
    get_localization_from_filename,
    CLASS2NAME,
    normalize_label,
    convert_hpa_labels_to_binary,
    get_subcell_class_names,
)
from evaluations.ca_esm import evaluate_fid as fid_module
from evaluations.ca_esm import evaluate_mse as mse_module
from evaluations.ca_esm import evaluate_intra_nuclear_prop as inp_module


# ---------------------------------------------------------------------------
# Protein sequence loading
# ---------------------------------------------------------------------------

def load_protein_sequences(cache_path):
    """Load cached protein sequences and sanitize to plain amino-acid strings."""
    with open(cache_path, "rb") as f:
        sequences = pickle.load(f)
    print(f"Loaded {len(sequences)} protein sequences from cache")

    # ESM tokenization expects only residue tokens, but cached values can include
    # FASTA headers such as '>sp|...'. Strip headers and non-residue characters.
    valid_chars = set("ACDEFGHIKLMNPQRSTVWYBXZOUJ-")
    sanitized_sequences = {}
    modified = 0
    empty_after_cleaning = 0

    for uniprot_id, seq in sequences.items():
        if not isinstance(seq, str):
            empty_after_cleaning += 1
            continue

        raw = seq.strip()
        if not raw:
            empty_after_cleaning += 1
            continue

        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if lines and lines[0].startswith(">"):
            lines = lines[1:]
        joined = "".join(lines).upper()

        # Keep only valid residue chars for model tokenization.
        cleaned = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYBXZOUJ-]", "", joined)

        if not cleaned:
            empty_after_cleaning += 1
            continue

        if cleaned != raw.upper():
            modified += 1
        if any(ch not in valid_chars for ch in cleaned):
            empty_after_cleaning += 1
            continue

        sanitized_sequences[uniprot_id] = cleaned

    print(
        f"Sanitized protein sequences: kept {len(sanitized_sequences)}, "
        f"modified {modified}, dropped {empty_after_cleaning}"
    )
    return sanitized_sequences


def get_uniprot_id(filename):
    """Extract UniProt ID from filename like Q13950_HPA022040_A-431_fov_1.npy"""
    return filename.replace(".npy", "").split("_")[0]


# ---------------------------------------------------------------------------
# TIFF loading and preprocessing for CELL-E2
# ---------------------------------------------------------------------------

def parse_npy_filename(filename):
    """Parse Q13950_HPA022040_A-431_fov_1.npy -> (uniprot, antibody, cell_line, fov)"""
    stem = filename.replace(".npy", "")
    prefix, fov = stem.rsplit("_fov_", 1)
    tokens = prefix.split("_")
    uniprot = tokens[0]
    antibody = tokens[1]
    cell_line = "_".join(tokens[2:])
    return uniprot, antibody, cell_line, fov


def get_tiff_dir(filename, tiff_root):
    """Map .npy filename to raw TIFF directory path."""
    uniprot, antibody, cell_line, fov = parse_npy_filename(filename)
    return os.path.join(tiff_root, uniprot, antibody, cell_line, f"fov_{fov}")


def load_tiffs(tiff_dir):
    """Load 4 TIFF channels from a FOV directory.

    Returns dict with keys: nucleus, microtubule, er, protein
    as float32 arrays in original uint16 range.
    """
    tiff_dir = Path(tiff_dir)
    all_tiffs = [p for p in tiff_dir.rglob("*") if p.suffix.lower() in {".tif", ".tiff"}]

    color_to_channel = {"blue": "nucleus", "red": "microtubule", "yellow": "er", "green": "protein"}
    channels = {}
    for color, channel_name in color_to_channel.items():
        matching = [p for p in all_tiffs if color.lower() in p.stem.lower()]
        if not matching:
            raise FileNotFoundError(f"No TIFF found for channel '{color}' in {tiff_dir}")

        tiff_path = matching[0]
        try:
            with Image.open(tiff_path) as img:
                channels[channel_name] = np.array(img).astype(np.float32)
        except Exception:
            channels[channel_name] = imread(str(tiff_path)).astype(np.float32)

    return channels


def preprocess_channel(img):
    """Clip at 0.001/99.999 percentiles and rescale to [0, 1]."""
    lower = np.percentile(img, 0.001)
    upper = np.percentile(img, 99.999)
    img = np.clip(img, lower, upper)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
    return img


def preprocess_tiffs_for_celle(tiff_dir):
    """Load raw TIFFs and preprocess for CELL-E2.

    Pipeline:
      1. Load TIFFs (4 channels)
      2. Percentile clip (0.001/99.999) + rescale to [0, 1] per channel
      3. Center crop to 1024x1024
      4. Resize to 256x256
      5. Apply CELL-E2 HPA z-score normalization on nucleus
      6. Final scaling: nucleus /= abs(max)

    Returns:
        nucleus_input: (1, 1, 256, 256) tensor for CELL-E2 input
        true_channels: (4, 256, 256) numpy array of all channels (pre-z-score, in [0,1])
    """
    raw = load_tiffs(tiff_dir)

    # Preprocess each channel: percentile clip + rescale to [0, 1]
    channel_names = ["nucleus", "microtubule", "er", "protein"]
    processed = {name: preprocess_channel(raw[name]) for name in channel_names}

    # Stack as [4, H, W] tensor
    stacked = torch.stack(
        [torch.from_numpy(processed[name]).float() for name in channel_names],
        dim=0,
    )  # [4, H, W]

    # Center crop to 1024x1024, then resize to 256x256
    t_forms = transforms.Compose([
        transforms.CenterCrop(1024),
        transforms.Resize(256, antialias=None),
    ])
    stacked = t_forms(stacked)  # [4, 256, 256]

    # Save pre-normalization copy for ground truth
    true_channels = stacked.numpy().copy()  # [4, 256, 256] in [0, 1]

    # Apply CELL-E2 HPA z-score normalization to nucleus+protein (indices 0, 3)
    nucleus = stacked[0:1]  # [1, 256, 256]
    normalize = transforms.Normalize(mean=(0.0655,), std=(0.1732,))
    nucleus = normalize(nucleus)

    # Final CELL-E2 scaling on nucleus
    abs_max = nucleus.abs().max()
    if abs_max > 0:
        nucleus = nucleus / abs_max
    nucleus = torch.nan_to_num(nucleus, 0.0, 1.0, 0.0)

    nucleus_input = nucleus.unsqueeze(0)  # [1, 1, 256, 256]

    return nucleus_input, true_channels


# ---------------------------------------------------------------------------
# CELL-E2 model loading
# ---------------------------------------------------------------------------

def _esm2_factory_for_dim(model_dim):
    """Return fair-esm pretrained factory name for a given embedding dim."""
    dim_to_factory = {
        320: "esm2_t6_8M_UR50D",
        480: "esm2_t12_35M_UR50D",
        640: "esm2_t30_150M_UR50D",
        1280: "esm2_t33_650M_UR50D",
        2560: "esm2_t36_3B_UR50D",
    }
    return dim_to_factory.get(int(model_dim), "esm2_t33_650M_UR50D")


def build_ckpt_with_esm2_text_emb(ckpt_path, model_dim, text_embedding, cache_root=None):
    """
    Ensure checkpoint contains CELL-E2 text encoder weights under
    `celle.text_emb.model.*` by copying from fair-esm pretrained ESM2 weights.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise KeyError(f"Checkpoint at {ckpt_path} does not contain 'state_dict'")

    state_dict = ckpt["state_dict"]
    existing_text_emb = sum(1 for k in state_dict if k.startswith("celle.text_emb.model."))
    if existing_text_emb > 0:
        print(f"Checkpoint already has {existing_text_emb} celle.text_emb.model.* keys")
        return ckpt_path

    if str(text_embedding).lower() != "esm2":
        raise ValueError(
            f"Unsupported text_embedding='{text_embedding}'. "
            "This benchmark patching currently supports only esm2."
        )

    if cache_root:
        torch_cache = os.path.join(cache_root, "torch_cache")
        os.makedirs(torch_cache, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", torch_cache)

    from esm import pretrained

    factory_name = _esm2_factory_for_dim(model_dim)
    print(f"Downloading/loading ESM2 encoder weights via pretrained.{factory_name}()")
    model_factory = getattr(pretrained, factory_name)
    esm_model, _ = model_factory()
    esm_state = esm_model.state_dict()

    inserted = 0
    for key, value in esm_state.items():
        target_key = f"celle.text_emb.model.{key}"
        if target_key not in state_dict:
            state_dict[target_key] = value.detach().cpu()
            inserted += 1

    if inserted == 0:
        raise RuntimeError("Failed to insert any ESM2 text_emb weights into checkpoint.")

    patched_path = str(
        Path(ckpt_path).with_name(
            f"{Path(ckpt_path).stem}_with_esm2_text_emb_dim{int(model_dim)}.ckpt"
        )
    )
    torch.save(ckpt, patched_path)
    print(f"Wrote patched checkpoint with {inserted} text_emb keys to {patched_path}")
    return patched_path


def load_celle_model(args, device):
    """Load CELL-E2 model from checkpoint."""
    celle_dir = os.path.abspath(args.celle_dir)
    sys.path.insert(0, celle_dir)

    from celle_main import instantiate_from_config
    from omegaconf import OmegaConf

    # Determine config paths
    model_config_path = os.path.join(celle_dir, args.model_config)
    config = OmegaConf.load(model_config_path)

    # Set checkpoint path
    if args.model_ckpt:
        ckpt_path = args.model_ckpt
    else:
        # Auto-download from HuggingFace
        from huggingface_hub import hf_hub_download
        print("Downloading CELL-E2 checkpoint from HuggingFace...")
        ckpt_path = hf_hub_download(
            repo_id="HuangLab/CELL-E_2_HPA_480",
            filename="model.ckpt",
            cache_dir=args.hf_cache_dir,
        )
        print(f"Downloaded checkpoint to {ckpt_path}")

    model_dim = int(config["model"]["params"]["dim"])
    text_embedding = str(config["model"]["params"].get("text_embedding", "esm2"))
    ckpt_path = build_ckpt_with_esm2_text_emb(
        ckpt_path=ckpt_path,
        model_dim=model_dim,
        text_embedding=text_embedding,
        cache_root=args.hf_cache_dir,
    )
    config["model"]["params"]["ckpt_path"] = ckpt_path
    # Set VQGAN paths to None (weights are embedded in checkpoint)
    config["model"]["params"]["condition_model_path"] = None
    config["model"]["params"]["vqgan_model_path"] = None

    # Set VQGAN config paths (needed for architecture instantiation)
    nucleus_vqgan_config = os.path.join(celle_dir, "configs", "HPA_nucleus_vqgan.yaml")
    threshold_vqgan_config = os.path.join(celle_dir, "configs", "HPA_threshold_vqgan.yaml")
    config["model"]["params"]["vqgan_config_path"] = threshold_vqgan_config
    config["model"]["params"]["condition_config_path"] = nucleus_vqgan_config

    print("Instantiating CELL-E2 model...")
    print("Model config:")
    print(config["model"])
    model = instantiate_from_config(config["model"]).to(device)
    model.eval()
    print("CELL-E2 model loaded successfully")

    return model


def create_tokenizer(celle_dir):
    """Create a CellLoader instance just for tokenize_sequence()."""
    sys.path.insert(0, os.path.abspath(celle_dir))
    from dataloader import CellLoader

    dataset = CellLoader(
        sequence_mode="embedding",
        vocab="esm2",
        split_key="val",
        crop_method="center",
        resize=600,
        crop_size=256,
        text_seq_len=1000,
        pad_mode="end",
        threshold="median",
    )
    return dataset


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load data ----
    val_dir = os.path.join(os.path.join(args.val_data_dir, args.val_split), '0000')
    all_npy_files = sorted([f for f in os.listdir(val_dir) if f.endswith(".npy")])
    print(f"Total .npy files in {args.val_split}: {len(all_npy_files)}")

    # Filter to samples with existing TIFF directories
    npy_files = [
        f for f in all_npy_files
        if os.path.isdir(get_tiff_dir(f, args.tiff_root))
    ]
    print(f"Files with existing TIFF directories: {len(npy_files)}")

    if args.num_samples is not None:
        npy_files = npy_files[: args.num_samples]
    print(f"Using {len(npy_files)} validation samples")

    # ---- Load HPA annotations ----
    print(f"Loading HPA annotations from {args.hpa_csv}")
    hpa_df = pd.read_csv(args.hpa_csv)
    hpa_lookup = build_localization_lookup(hpa_df)

    # ---- Setup output directories ----
    output_dir = Path(args.output_dir)
    images_dir_true = output_dir / "images_true"
    images_dir_gen = output_dir / "images_gen"
    os.makedirs(images_dir_true, exist_ok=True)
    os.makedirs(images_dir_gen, exist_ok=True)

    # ---- Check if all images have already been generated ----
    all_generated = True
    expected_filenames = []
    for npy_file in npy_files:
        base_name = npy_file.replace(".npy", "")
        for prefix, img_dir in [("true", images_dir_true), ("gen", images_dir_gen)]:
            sample_id = f"{prefix}_{base_name}"
            for suffix in ["nucleus", "microtubule", "er", "protein"]:
                if not (img_dir / f"{sample_id}_{suffix}.png").exists():
                    all_generated = False
                    break
            if not all_generated:
                break
        if not all_generated:
            break
        expected_filenames.append(base_name)

    if all_generated:
        print(f"\nAll {len(expected_filenames)} samples already generated. Skipping generation.")
        all_filenames = expected_filenames
        skipped = 0
        # Reconstruct records from existing files
        records_true = []
        records_gen = []
        for base_name in all_filenames:
            for prefix, img_dir, records_list in [
                ("true", images_dir_true, records_true),
                ("gen", images_dir_gen, records_gen),
            ]:
                sample_id = f"{prefix}_{base_name}"
                records_list.append({
                    'r_image': str((img_dir / f"{sample_id}_microtubule.png").absolute()),
                    'y_image': str((img_dir / f"{sample_id}_er.png").absolute()),
                    'b_image': str((img_dir / f"{sample_id}_nucleus.png").absolute()),
                    'g_image': str((img_dir / f"{sample_id}_protein.png").absolute()),
                    'output_prefix': sample_id,
                })
    else:
        # ---- Load protein sequences ----
        protein_sequences = load_protein_sequences(args.protein_cache)

        # ---- Load model ----
        model = load_celle_model(args, device)
        tokenizer = create_tokenizer(args.celle_dir)

        # ---- Generation + saving loop ----
        records_true = []
        records_gen = []
        all_filenames = []
        skipped = 0

        print(f"\nGenerating protein channel images...")
        for i, npy_file in enumerate(tqdm(npy_files, desc="Generating")):
            # Get protein sequence
            uniprot_id = get_uniprot_id(npy_file)
            if uniprot_id not in protein_sequences:
                print(f"Skipping {npy_file}: no protein sequence for {uniprot_id}")
                skipped += 1
                continue
            protein_seq = protein_sequences[uniprot_id]
            if not protein_seq or len(protein_seq) == 0:
                print(f"Skipping {npy_file}: empty protein sequence for {uniprot_id}")
                skipped += 1
                continue

            # Load raw TIFFs and preprocess
            tiff_dir = get_tiff_dir(npy_file, args.tiff_root)
            try:
                nucleus_tensor, true_channels = preprocess_tiffs_for_celle(tiff_dir)
            except (FileNotFoundError, Exception) as e:
                print(f"Skipping {npy_file}: TIFF loading failed: {e}")
                skipped += 1
                continue

            nucleus_tensor = nucleus_tensor.to(device)

            # Tokenize protein sequence
            sequence_tensor = tokenizer.tokenize_sequence(protein_seq).to(device)

            # Generate protein channel
            with torch.no_grad():
                _, _, _, predicted_threshold, predicted_heatmap = model.celle.sample(
                    text=sequence_tensor,
                    condition=nucleus_tensor,
                    timesteps=1,
                    temperature=1.0,
                    filter_thres=0.9,
                    progress=False,
                )

            # Extract generated protein channel: [1, 1, 256, 256] -> [256, 256]
            gen_protein = predicted_heatmap.cpu().numpy()[0, 0]  # continuous heatmap
            # gen_protein = predicted_threshold.cpu().numpy()[0, 0]
            gen_protein = np.clip(gen_protein, 0, 1)

            # Build "generated" image: nucleus/MT/ER from raw + protein from CELL-E2
            gen_channels = true_channels.copy()  # [4, 256, 256]
            gen_channels[3] = gen_protein  # replace protein channel

            # Save images as (1, 4, H, W) batch
            base_name = npy_file.replace(".npy", "")
            true_batch = true_channels[np.newaxis]  # [1, 4, 256, 256]
            gen_batch = gen_channels[np.newaxis]  # [1, 4, 256, 256]

            true_recs = save_fov_images_for_subcell(
                true_batch, [base_name], str(images_dir_true), prefix="true"
            )
            gen_recs = save_fov_images_for_subcell(
                gen_batch, [base_name], str(images_dir_gen), prefix="gen"
            )

            records_true.extend(true_recs)
            records_gen.extend(gen_recs)
            all_filenames.append(base_name)

        print(f"\nGenerated {len(all_filenames)} samples, skipped {skipped}")

    if len(all_filenames) == 0:
        print("ERROR: No samples generated. Exiting.")
        return

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark CELL-E2 on protein localization evaluation"
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default="/n/netscratch/chenf2011_lab/sandeep/data/hpa_dataset/hpa_preprocessed_split_256/concatenated_arrays",  # change to your path
        help="Path to .npy validation images directory (used for filename listing)",
    )
    parser.add_argument(
        "--tiff_root",
        type=str,
        default="/n/netscratch/chenf2011_lab/sandeep/data/hpa_dataset/humanproteinatlas_temp",  # change to your path
        help="Path to raw HPA TIFF download directory",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="val1",
        help="Validation split name (e.g., val1, val2, cell_line_holdouts)",
    )
    parser.add_argument(
        "--celle_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "CELL-E_2"),
        help="Path to CELL-E2 codebase",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/HPA_celle_ESM2_480.yaml",
        help="Model config yaml (relative to celle_dir)",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=None,
        help="Path to model checkpoint (auto-downloads from HF if not set)",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default="/n/netscratch/chenf2011_lab/sandeep",  # change to your path
        help="Hugging Face cache directory used for auto-downloaded model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/n/netscratch/chenf2011_lab/sandeep/evaluation_results_benchmarks/CELL-E2_HPA_480",  # change to your path
        help="Output directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation (CELL-E2 is memory-intensive)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all)",
    )
    parser.add_argument(
        "--hpa_csv",
        type=str,
        default="/n/home08/skambhampati/ar-microscopy-gen/code/data_preparation/hpa_cell-line-download.csv",  # change to your path
        help="HPA annotation CSV",
    )
    parser.add_argument(
        "--subcell_dir",
        type=str,
        default=os.path.expanduser("~/SubCellPortable"),
        help="Path to SubCellPortable",
    )
    parser.add_argument(
        "--protein_cache",
        type=str,
        default="/n/home08/skambhampati/ar-microscopy-gen/code/data_preparation/.protein_cache/protein_sequences.pkl",  # change to your path
        help="Path to cached protein sequences pkl",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        default=False,
        help="Skip evaluation (default: False)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("CELL-E2 Protein Localization Benchmark")
    print("=" * 60)
    print(f"Val data dir: {args.val_data_dir}")
    print(f"TIFF root: {args.tiff_root}")
    print(f"Val split: {args.val_split}")
    print(f"CELL-E2 dir: {args.celle_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Num samples: {args.num_samples or 'all'}")
    print("=" * 60 + "\n")

    evaluate(args)


if __name__ == "__main__":
    main()
