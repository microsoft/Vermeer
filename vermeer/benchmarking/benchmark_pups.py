#!/usr/bin/env python3
# Code derived from original PUPS repo (https://github.com/uhlerlab/PUPS)
"""
Benchmark PUPS on the same protein localization evaluation used for CA_LlamaGen.

Generates protein channel images from raw HPA TIFFs using PUPS (protein channel
prediction from landmark stains + ESM2 embeddings), then runs SubCellPortable
to classify protein localization, comparing predictions against HPA ground truth.

Pipeline (parallelized):
    Phase 0: Pre-compute ESM2 embeddings for all unique proteins (cached to disk)
    Phase 1: Parallel CPU preprocessing — 4 workers load TIFFs, segment nuclei,
             extract 128x128 crops, save .npz files to intermediate directory
    Phase 2: Batched GPU inference — read all crops, batch through PUPS model,
             save true/generated PNGs
    Phase 3: Evaluate — SubCellPortable localization, FID, MSE, intra-nuclear proportion

Usage:
    python benchmark_pups.py \\
        --val_dir /path/to/code_directory \\
        --val_split val1 \\
        --raw_hpa_dir /path/to/humanproteinatlas_temp \\
        --output_dir /path/to/output \\
        --num_workers 4
"""

import numpy as np
from pathlib import Path
import argparse
import json
import torch
import sys
import os
import pickle
import re
import gc
import h5py
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from PIL import Image
from tifffile import imread as tifffile_imread
import pandas as pd
import ast
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
CA_LLAMAGEN_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(CA_LLAMAGEN_DIR))
from evaluations.ca_esm.evaluation_protein_loc import (
    save_fov_images_for_subcell,
    build_localization_lookup
)

# from evaluations.ca_esm.evaluation_protein_loc import (
#     save_fov_images_for_subcell,
#     create_subcell_csv,
#     run_subcell_portable,
#     parse_subcell_results,
#     parse_subcell_probabilities,
#     compute_metrics_for_group,
#     build_localization_lookup,
#     get_localization_from_filename,
#     CLASS2NAME,
#     normalize_label,
#     convert_hpa_labels_to_binary,
#     get_subcell_class_names,
# )

# from evaluations.ca_esm import evaluate_fid as fid_module
# from evaluations.ca_esm import evaluate_mse as mse_module
# from evaluations.ca_esm import evaluate_intra_nuclear_prop as inp_module


# ---------------------------------------------------------------------------
# Protein sequence loading (same pattern as benchmark_celle2.py)
# ---------------------------------------------------------------------------

def load_protein_sequences(cache_path):
    """Load cached protein sequences and sanitize to plain amino-acid strings."""
    with open(cache_path, "rb") as f:
        sequences = pickle.load(f)
    print(f"Loaded {len(sequences)} protein sequences from cache")

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


def parse_filename(filename):
    """
    Parse validation filename into components.

    Q13950_HPA022040_A-431_fov_1.npy
    → uniprot_id='Q13950', antibody='HPA022040', cell_line='A-431', fov='1'
    """
    stem = filename.replace(".npy", "")
    base, fov_num = stem.rsplit("_fov_", 1)
    parts = base.split("_", 2)
    return {
        "uniprot_id": parts[0],
        "antibody": parts[1],
        "cell_line": parts[2],
        "fov": fov_num,
    }


def handle_special_cell_line_name(cell_line):
    """Handle cell line names that contain '/' in the directory structure but '_' in filenames."""
    if "RPTEC_TERT1" in cell_line:
        return cell_line.replace("RPTEC_TERT1", "RPTEC/TERT1")
    if "HUVEC_TERT2" in cell_line:
        return cell_line.replace("HUVEC_TERT2", "HUVEC/TERT2")
    if "PODO_TERT256" in cell_line:
        return cell_line.replace("PODO_TERT256", "PODO/TERT256")
    if "PODO_TERT25" in cell_line:
        return cell_line.replace("PODO_TERT25", "PODO/TERT25")
    if "PODO_SVTERT152" in cell_line:
        return cell_line.replace("PODO_SVTERT152", "PODO/SVTERT152")
    return cell_line


def build_raw_hpa_path(raw_hpa_dir, parsed):
    """Build path to raw HPA TIFF directory for a parsed filename."""
    cell_line = handle_special_cell_line_name(parsed["cell_line"])
    return os.path.join(
        raw_hpa_dir,
        parsed["uniprot_id"],
        parsed["antibody"],
        cell_line,
        f"fov_{parsed['fov']}",
    )


# ---------------------------------------------------------------------------
# PUPS preprocessing
# ---------------------------------------------------------------------------

def load_tiff_channels(fov_dir):
    """
    Load 4 TIFF channels (red, green, blue, yellow) from a FOV directory.
    Returns dict of channel_name → float32 array.
    """
    fov_path = Path(fov_dir)
    all_tiffs = [p for p in fov_path.rglob("*") if p.suffix.lower() in {".tif", ".tiff"}]

    channel_order = ["red", "green", "blue", "yellow"]
    raw_channels = {}
    for color in channel_order:
        matching = [p for p in all_tiffs if color.lower() in p.stem.lower()]
        if not matching:
            raise FileNotFoundError(f"No TIFF found for channel '{color}' in {fov_dir}")
        try:
            img = np.array(Image.open(matching[0])).astype(np.float32)
        except Exception:
            try:
                img = tifffile_imread(matching[0]).astype(np.float32)
            except Exception as e2:
                raise Exception(f"Failed to read image with both PIL and tifffile: {e2}")
        raw_channels[color] = img

    return raw_channels


def preprocess_fov(raw_channels, perform_unmixing=False):
    """
    Full PUPS preprocessing pipeline on raw TIFF channels.

    Returns list of crop dicts with keys:
        nuclei, microtubules, antibody, mitochondria  (each 128x128 float32)
    """
    from skimage.transform import resize
    from skimage.filters import threshold_otsu, gaussian
    from skimage.measure import label
    from skimage.segmentation import clear_border
    from skimage.morphology import remove_small_objects, remove_small_holes
    from skimage.exposure import rescale_intensity
    from scipy.ndimage import center_of_mass

    # Assign channels
    if perform_unmixing:
        from scipy.linalg import lstsq

        rgb_composite = np.stack([
            raw_channels["red"],
            raw_channels["green"],
            raw_channels["blue"],
        ], axis=-1)

        spectral_sigs = _get_stain_signatures(rgb_composite)
        flat = rgb_composite.reshape(-1, 3).T
        fractions, _, _, _ = lstsq(spectral_sigs.T, flat)
        fractions = np.clip(fractions, 0, 1)
        orig_dim = int(np.sqrt(fractions.shape[1]))
        unmixed = fractions.reshape(fractions.shape[0], orig_dim, orig_dim)

        channel_microtubules = rescale_intensity(unmixed[0].astype(np.float32), out_range=(0, 1))
        channel_antibody = rescale_intensity(unmixed[1].astype(np.float32), out_range=(0, 1))
        channel_nuclei = rescale_intensity(unmixed[2].astype(np.float32), out_range=(0, 1))
        channel_mitochondria = rescale_intensity(
            raw_channels["yellow"].astype(np.float32), out_range=(0, 1)
        )
    else:
        channel_microtubules = raw_channels["red"]
        channel_antibody = raw_channels["green"]
        channel_nuclei = raw_channels["blue"]
        channel_mitochondria = raw_channels["yellow"]

    # Downsample 4x
    DOWNSAMPLE = 4
    h, w = channel_nuclei.shape[:2]
    new_h, new_w = h // DOWNSAMPLE, w // DOWNSAMPLE

    nuclei_stain = resize(channel_nuclei, (new_h, new_w), preserve_range=True).astype(np.float32)
    microtubules_stain = resize(channel_microtubules, (new_h, new_w), preserve_range=True).astype(np.float32)
    antibody_stain = resize(channel_antibody, (new_h, new_w), preserve_range=True).astype(np.float32)
    mitochondria_stain = resize(channel_mitochondria, (new_h, new_w), preserve_range=True).astype(np.float32)

    # Nuclear segmentation
    MIN_NUC_SIZE = 100
    val = threshold_otsu(nuclei_stain)
    smoothed_nuclei = gaussian(nuclei_stain, sigma=5.0)
    binary_nuclei = smoothed_nuclei > val
    binary_nuclei = remove_small_holes(binary_nuclei, area_threshold=300)
    labeled_nuclei = label(binary_nuclei)
    labeled_nuclei = clear_border(labeled_nuclei)
    labeled_nuclei = remove_small_objects(labeled_nuclei, min_size=MIN_NUC_SIZE)

    n_nuclei = np.max(labeled_nuclei)

    # 128x128 single-cell crops
    CROP_SIZE = 128
    FILTER_THRESH = 0.19

    crops = []
    for i in range(1, n_nuclei + 1):
        current_nuc = labeled_nuclei == i
        if np.sum(current_nuc) <= MIN_NUC_SIZE:
            continue

        y, x = center_of_mass(current_nuc)
        x, y = int(x), int(y)

        c1 = y - CROP_SIZE // 2
        c2 = y + CROP_SIZE // 2
        c3 = x - CROP_SIZE // 2
        c4 = x + CROP_SIZE // 2

        if c1 < 0 or c3 < 0 or c2 > new_h or c4 > new_w:
            continue

        nuc_crop = rescale_intensity(nuclei_stain[c1:c2, c3:c4].astype(np.float32), out_range=(0, 1))
        mt_crop = rescale_intensity(microtubules_stain[c1:c2, c3:c4].astype(np.float32), out_range=(0, 1))
        ab_crop = rescale_intensity(antibody_stain[c1:c2, c3:c4].astype(np.float32), out_range=(0, 1))
        mito_crop = rescale_intensity(mitochondria_stain[c1:c2, c3:c4].astype(np.float32), out_range=(0, 1))

        # Save un-thresholded landmarks for SubCell (before PUPS thresholding)
        nuc_crop_raw = nuc_crop.copy()
        mt_crop_raw = mt_crop.copy()
        mito_crop_raw = mito_crop.copy()

        # Apply 0.19 low-value threshold to landmark stains (required by PUPS)
        nuc_crop[nuc_crop < FILTER_THRESH] = 0.0
        mt_crop[mt_crop < FILTER_THRESH] = 0.0
        mito_crop[mito_crop < FILTER_THRESH] = 0.0

        crops.append({
            "nuclei": nuc_crop,
            "microtubules": mt_crop,
            "antibody": ab_crop,
            "mitochondria": mito_crop,
            "nuclei_raw": nuc_crop_raw,
            "microtubules_raw": mt_crop_raw,
            "mitochondria_raw": mito_crop_raw,
        })

    return crops


def _get_stain_signatures(image):
    """Compute stain spectral signatures from RGB composite (for unmixing)."""
    import cv2

    hsv_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

    red_lower_1 = np.array([170, 30, 30])
    red_upper_1 = np.array([179, 150, 150])
    red_lower_2 = np.array([0, 30, 30])
    red_upper_2 = np.array([10, 150, 150])
    blue_lower = np.array([110, 30, 30])
    blue_upper = np.array([130, 150, 150])
    green_lower = np.array([50, 30, 30])
    green_upper = np.array([70, 150, 150])
    yellow_lower = np.array([20, 30, 30])
    yellow_upper = np.array([40, 150, 150])

    red_mask = cv2.inRange(hsv_image, red_lower_1, red_upper_1) | cv2.inRange(
        hsv_image, red_lower_2, red_upper_2
    )
    blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

    def _mean_sig(mask):
        pixels = hsv_image[np.where(mask != 0)]
        if len(pixels) == 0:
            return np.array([0.0, 0.0, 0.0])
        rgb = cv2.cvtColor(np.uint8([pixels]), cv2.COLOR_HSV2RGB).mean(axis=1)[0] / 255
        mx = rgb.max()
        return rgb / mx if mx > 0 else rgb

    return np.array([
        _mean_sig(red_mask),
        _mean_sig(green_mask),
        _mean_sig(blue_mask),
        _mean_sig(yellow_mask),
    ])


# ---------------------------------------------------------------------------
# PUPS model loading
# ---------------------------------------------------------------------------

def load_pups_model(pups_ckpt, pups_dir, device="cpu"):
    """Load PUPS model from checkpoint."""
    sys.path.insert(0, str(pups_dir))
    from src.model.full_model import SubCellProtModel

    model = SubCellProtModel.load_from_checkpoint(pups_ckpt, map_location="cpu")
    model = model.to(device)
    model.eval()
    print(f"PUPS model loaded successfully on {device}")
    return model


def load_esm2_utils(pups_dir):
    """Import get_esm2_representation from PUPS."""
    sys.path.insert(0, str(pups_dir))
    from src.utils.esm2_utils import get_esm2_representation
    return get_esm2_representation


# ---------------------------------------------------------------------------
# UniProt sequence fetching (fallback when not in cache)
# ---------------------------------------------------------------------------

def fetch_uniprot_sequence(uniprot_id, timeout=30):
    """Fetch amino acid sequence from UniProt REST FASTA endpoint."""
    from urllib.request import urlopen
    from urllib.error import HTTPError, URLError

    fasta_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        with urlopen(fasta_url, timeout=timeout) as response:
            fasta_text = response.read().decode("utf-8")
    except HTTPError as e:
        raise RuntimeError(
            f"UniProt request failed for '{uniprot_id}' with HTTP {e.code}"
        ) from e
    except URLError as e:
        raise RuntimeError(
            f"Network error fetching UniProt sequence for '{uniprot_id}': {e}"
        ) from e

    lines = [line.strip() for line in fasta_text.splitlines() if line.strip()]
    if len(lines) < 2 or not lines[0].startswith(">"):
        raise RuntimeError(f"Unexpected FASTA response for '{uniprot_id}'")

    sequence = "".join(lines[1:]).upper()
    valid_aa = set("ACDEFGHIKLMNPQRSTVWYBXZOUJ*")
    invalid = sorted({ch for ch in sequence if ch not in valid_aa})
    if invalid:
        raise RuntimeError(f"Sequence for '{uniprot_id}' has unexpected residues: {invalid}")
    if len(sequence) == 0:
        raise RuntimeError(f"Empty UniProt sequence for '{uniprot_id}'")

    return sequence


# ---------------------------------------------------------------------------
# Phase 0: Pre-compute ESM2 embeddings
# ---------------------------------------------------------------------------

def precompute_esm2_cache(npy_files, protein_sequences, esm2_cache_path, pups_dir):
    """
    Pre-compute ESM2 embeddings for all unique UniProt IDs, streaming to HDF5.
    Skips if cache file already exists.
    Returns the path to the HDF5 cache file.
    """

    if os.path.exists(esm2_cache_path):
        with h5py.File(esm2_cache_path, "r") as f:
            n_cached = len(f["X_esm2"])
        print(f"ESM2 cache exists at {esm2_cache_path} ({n_cached} embeddings)")
        return esm2_cache_path

    # Collect unique UniProt IDs
    unique_ids = sorted(set(get_uniprot_id(f) for f in npy_files))
    print(f"\nPhase 0: Pre-computing ESM2 embeddings for {len(unique_ids)} unique proteins")

    get_esm2_representation = load_esm2_utils(pups_dir)

    skipped = 0
    computed = 0
    with h5py.File(esm2_cache_path, "w") as f:
        x_grp = f.create_group("X_esm2")
        xlen_grp = f.create_group("x_len")

        for uid in tqdm(unique_ids, desc="ESM2 embeddings"):
            if uid not in protein_sequences:
                try:
                    seq = fetch_uniprot_sequence(uid)
                    protein_sequences[uid] = seq
                except Exception as e:
                    print(f"  WARNING: No sequence for {uid}: {e}")
                    skipped += 1
                    continue

            seq = protein_sequences[uid]
            if not seq or len(seq) == 0:
                print(f"  WARNING: Empty sequence for {uid}")
                skipped += 1
                continue

            try:
                X_esm2, x_len = get_esm2_representation(uid, seq)
                # Stream to disk immediately — don't accumulate in memory
                x_grp.create_dataset(uid, data=X_esm2.numpy())
                xlen_grp.create_dataset(uid, data=x_len.numpy())
                computed += 1
            except Exception as e:
                print(f"  WARNING: ESM2 failed for {uid}: {e}")
                skipped += 1

    print(f"  Computed {computed} embeddings, skipped {skipped}")
    print(f"  Saved ESM2 cache to {esm2_cache_path}")

    # Free ESM2 model from memory before spawning workers
    if "src.utils.esm2_utils" in sys.modules:
        esm2_mod = sys.modules["src.utils.esm2_utils"]
        if hasattr(esm2_mod, "model"):
            del esm2_mod.model
        del sys.modules["src.utils.esm2_utils"]
    gc.collect()

    return esm2_cache_path


# ---------------------------------------------------------------------------
# Phase 1: Parallel CPU preprocessing
# ---------------------------------------------------------------------------

def _preprocess_worker(args_tuple):
    """
    Worker function for multiprocessing. Loads TIFFs, runs preprocessing,
    saves crops as .npz to intermediate directory.

    Returns: (npy_file, n_crops, error_or_None)
    """
    npy_file, raw_hpa_dir, perform_unmixing, intermediate_dir = args_tuple

    stem = npy_file.replace(".npy", "")
    npz_path = os.path.join(intermediate_dir, f"{stem}.npz")

    # Skip if already processed
    if os.path.exists(npz_path):
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                meta = json.loads(str(data["metadata"]))
                return (npy_file, meta["n_crops"], None)
        except Exception:
            pass  # Re-process if corrupt

    parsed = parse_filename(npy_file)
    uniprot_id = parsed["uniprot_id"]

    raw_path = build_raw_hpa_path(raw_hpa_dir, parsed)
    if not os.path.isdir(raw_path):
        return (npy_file, 0, f"Raw HPA dir not found: {raw_path}")

    try:
        raw_channels = load_tiff_channels(raw_path)
    except FileNotFoundError as e:
        return (npy_file, 0, str(e))

    try:
        crops = preprocess_fov(raw_channels, perform_unmixing=perform_unmixing)
    except Exception as e:
        return (npy_file, 0, f"Preprocessing failed: {e}")

    if len(crops) == 0:
        return (npy_file, 0, "No valid crops")

    # Save crops as .npz: crop_000=[4,128,128] (thresholded, for PUPS),
    # crop_raw_000=[3,128,128] (un-thresholded landmarks, for SubCell), metadata=json
    arrays = {}
    for i, crop in enumerate(crops):
        stacked = np.stack([
            crop["nuclei"],
            crop["microtubules"],
            crop["mitochondria"],
            crop["antibody"],
        ], axis=0)  # [4, 128, 128]
        arrays[f"crop_{i:03d}"] = stacked
        raw_landmarks = np.stack([
            crop["nuclei_raw"],
            crop["microtubules_raw"],
            crop["mitochondria_raw"],
        ], axis=0)  # [3, 128, 128]
        arrays[f"crop_raw_{i:03d}"] = raw_landmarks

    metadata = {
        "source_file": npy_file,
        "uniprot_id": uniprot_id,
        "n_crops": len(crops),
    }
    arrays["metadata"] = np.array(json.dumps(metadata))

    np.savez(npz_path, **arrays)
    return (npy_file, len(crops), None)


def run_parallel_preprocessing(npy_files, raw_hpa_dir, perform_unmixing, intermediate_dir, num_workers):
    """
    Phase 1: Run preprocessing across FOVs using multiprocessing.
    Returns preprocess_manifest (list of dicts) and skip count.
    """
    os.makedirs(intermediate_dir, exist_ok=True)

    args_list = [
        (f, raw_hpa_dir, perform_unmixing, intermediate_dir)
        for f in npy_files
    ]

    print(f"\nPhase 1: Parallel preprocessing with {num_workers} workers")
    print(f"  {len(npy_files)} FOVs → {intermediate_dir}")

    # Limit OpenBLAS threads to avoid fork issues
    old_omp = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"

    manifest = []
    skipped = 0

    with multiprocessing.Pool(num_workers) as pool:
        for npy_file, n_crops, error in tqdm(
            pool.imap_unordered(_preprocess_worker, args_list),
            total=len(args_list),
            desc="Preprocessing FOVs",
        ):
            if error is not None:
                print(f"  WARNING: {npy_file}: {error}")
                skipped += 1
                continue
            if n_crops == 0:
                skipped += 1
                continue

            stem = npy_file.replace(".npy", "")
            manifest.append({
                "file": npy_file,
                "uniprot_id": get_uniprot_id(npy_file),
                "n_crops": n_crops,
                "npz_path": os.path.join(intermediate_dir, f"{stem}.npz"),
            })

    # Restore OMP_NUM_THREADS
    if old_omp is not None:
        os.environ["OMP_NUM_THREADS"] = old_omp
    elif "OMP_NUM_THREADS" in os.environ:
        del os.environ["OMP_NUM_THREADS"]

    total_crops = sum(e["n_crops"] for e in manifest)
    print(f"  Preprocessed {len(manifest)} FOVs ({total_crops} crops), skipped {skipped}")

    return manifest, skipped


# ---------------------------------------------------------------------------
# Phase 2: Batched GPU inference
# ---------------------------------------------------------------------------

def run_batched_inference(
    preprocess_manifest, esm2_cache_path, pups_ckpt, pups_dir,
    images_dir_true, images_dir_gen, crop_batch_size, device="cpu",
):
    """
    Phase 2: Load all intermediate crops, batch through PUPS, save PNGs.
    Reads ESM2 embeddings on-demand from HDF5 to avoid holding all in memory.
    Returns (records_true, records_gen, all_crop_names, all_filenames_for_crops).
    """
    print(f"\nPhase 2: Batched GPU inference (batch_size={crop_batch_size})")

    # Load PUPS model
    model = load_pups_model(pups_ckpt, pups_dir, device=device)

    os.makedirs(str(images_dir_true), exist_ok=True)
    os.makedirs(str(images_dir_gen), exist_ok=True)

    # Collect all crops with metadata into a flat list
    # Each entry: (thresholded[4,128,128], raw_landmarks[3,128,128], crop_name, source_file, uniprot_id)

    print("  Loading intermediate crops...")
    crop_entries = []
    for entry in tqdm(preprocess_manifest, desc="  Loading .npz files"):
        npz_path = entry["npz_path"]
        npy_file = entry["file"]
        uniprot_id = entry["uniprot_id"]
        n_crops = entry["n_crops"]
        base_name = npy_file.replace(".npy", "")

        with np.load(npz_path, allow_pickle=True) as data:
            for i in range(n_crops):
                arr = data[f"crop_{i:03d}"]  # [4, 128, 128]: nuclei, mt, mito, antibody (thresholded landmarks)
                raw_key = f"crop_raw_{i:03d}"
                if raw_key in data:
                    arr_raw = data[raw_key]  # [3, 128, 128]: nuclei, mt, mito (un-thresholded)
                else:
                    arr_raw = arr[:3]  # fallback for old npz files without raw landmarks
                crop_name = f"{base_name}_crop{i:03d}"
                crop_entries.append((arr, arr_raw, crop_name, npy_file, uniprot_id))

    print(f"  Total crops to process: {len(crop_entries)}")

    # Process in batches
    records_true = []
    records_gen = []
    all_crop_names = []
    all_filenames_for_crops = []

    # Thread pool for async PNG saving
    io_executor = ThreadPoolExecutor(max_workers=4)
    io_futures = []

    # Open HDF5 cache for on-demand reading (kept open for the loop)
    esm2_h5 = h5py.File(esm2_cache_path, "r")
    x_grp = esm2_h5["X_esm2"]
    xlen_grp = esm2_h5["x_len"]

    for batch_start in tqdm(range(0, len(crop_entries), crop_batch_size), desc="  PUPS inference"):
        batch = crop_entries[batch_start : batch_start + crop_batch_size]
        n_batch = len(batch)

        # Stack landmarks [N, 3, 128, 128] (channels 0-2: nuclei, mt, mito) — thresholded for PUPS
        landmark_batch = torch.from_numpy(
            np.stack([e[0][:3] for e in batch])
        ).float()  # [N, 3, 128, 128]

        # Build ESM2 batch [N, 1, 2000, 1280] — read from HDF5 on demand
        esm2_tensors = []
        x_lens = []
        for _, _, _, _, uid in batch:
            if uid in x_grp:
                X_esm2 = torch.from_numpy(x_grp[uid][:])
                x_len = torch.from_numpy(xlen_grp[uid][:])
                esm2_tensors.append(X_esm2.unsqueeze(1))  # [1, 1, 2000, 1280]
                x_lens.append(x_len)
            else:
                # Should not happen if Phase 0 ran, but handle gracefully
                esm2_tensors.append(torch.zeros(1, 1, 2000, 1280))
                x_lens.append(torch.tensor([0]))

        X_esm2_batch = torch.cat(esm2_tensors, dim=0)  # [N, 1, 2000, 1280]
        x_len_batch = torch.cat([xl.view(-1) for xl in x_lens], dim=0)  # [N]

        with torch.no_grad():
            pred_protein_batch, _ = model.call_model(
                X_esm2_batch.to(device),
                x_len_batch.to(device),
                landmark_batch.to(device),
            )

        gen_protein_np = pred_protein_batch.cpu().numpy()
        if gen_protein_np.ndim == 4:
            gen_protein_np = gen_protein_np[:, 0]  # [N, 128, 128]
        gen_protein_np = np.clip(gen_protein_np, 0, 1)

        # Save images (async I/O) — use un-thresholded landmarks for SubCell compatibility
        for j in range(n_batch):
            arr, arr_raw, crop_name, npy_file, uid = batch[j]
            gen_protein = gen_protein_np[j]

            # Use raw (un-thresholded) landmarks + original antibody for saved images
            true_channels = np.stack([
                arr_raw[0], arr_raw[1], arr_raw[2],  # nuclei, mt, mito (un-thresholded)
                arr[3],  # antibody (was never thresholded)
            ], axis=0)[np.newaxis]  # [1, 4, 128, 128]
            gen_channels = np.stack([
                arr_raw[0], arr_raw[1], arr_raw[2],  # nuclei, mt, mito (un-thresholded)
                gen_protein,
            ], axis=0)[np.newaxis]  # [1, 4, 128, 128]

            all_crop_names.append(crop_name)
            all_filenames_for_crops.append(npy_file)

            # Submit PNG saving to thread pool
            def _save_images(tc, gc, cn, itrue, igen):
                true_recs = save_fov_images_for_subcell(tc, [cn], str(itrue), prefix="true")
                gen_recs = save_fov_images_for_subcell(gc, [cn], str(igen), prefix="gen")
                return true_recs, gen_recs

            fut = io_executor.submit(_save_images, true_channels, gen_channels, crop_name, images_dir_true, images_dir_gen)
            io_futures.append(fut)

    # Collect I/O results
    print("  Waiting for PNG saves to complete...")
    for fut in tqdm(io_futures, desc="  Collecting I/O"):
        true_recs, gen_recs = fut.result()
        records_true.extend(true_recs)
        records_gen.extend(gen_recs)

    io_executor.shutdown(wait=True)
    esm2_h5.close()

    print(f"  Saved {len(all_crop_names)} crop pairs")
    return records_true, records_gen, all_crop_names, all_filenames_for_crops


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Enumerate validation samples ----
    val_code_dir = os.path.join(args.val_dir, f"ca256_codes", args.val_split)
    npy_files = sorted([f for f in os.listdir(val_code_dir) if f.endswith(".npy")])
    if args.num_samples is not None:
        npy_files = npy_files[: args.num_samples]
    print(f"Found {len(npy_files)} validation samples in {val_code_dir}")

    # ---- Load HPA annotations ----
    print(f"Loading HPA annotations from {args.hpa_csv}")
    hpa_df = pd.read_csv(args.hpa_csv)
    hpa_lookup = build_localization_lookup(hpa_df)

    # ---- Setup output directories ----
    output_dir = Path(args.output_dir)
    images_dir_true = output_dir / "images_true"
    images_dir_gen = output_dir / "images_gen"
    intermediate_dir = str(output_dir / "intermediate_crops")
    esm2_cache_path = str(output_dir / "esm2_cache.h5")
    manifest_path = output_dir / "crop_manifest.json"
    preprocess_manifest_path = output_dir / "preprocess_manifest.json"
    os.makedirs(images_dir_true, exist_ok=True)
    os.makedirs(images_dir_gen, exist_ok=True)

    # ---- Check if all images have already been generated ----
    all_generated = False
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        all_crop_names = manifest["crop_names"]
        all_filenames_for_crops = manifest["source_filenames"]
        # Verify all files exist
        all_generated = True
        for crop_name in all_crop_names:
            for prefix, img_dir in [("true", images_dir_true), ("gen", images_dir_gen)]:
                sample_id = f"{prefix}_{crop_name}"
                for suffix in ["nucleus", "microtubule", "er", "protein"]:
                    if not (img_dir / f"{sample_id}_{suffix}.png").exists():
                        all_generated = False
                        break
                if not all_generated:
                    break
            if not all_generated:
                break

    if all_generated and manifest_path.exists():
        print(f"\nAll {len(all_crop_names)} crops already generated. Skipping generation.")
        skipped = manifest.get("skipped", 0)

        # Reconstruct records
        records_true = []
        records_gen = []
        for crop_name in all_crop_names:
            for prefix, img_dir, records_list in [
                ("true", images_dir_true, records_true),
                ("gen", images_dir_gen, records_gen),
            ]:
                sample_id = f"{prefix}_{crop_name}"
                records_list.append({
                    "r_image": str((img_dir / f"{sample_id}_microtubule.png").absolute()),
                    "y_image": str((img_dir / f"{sample_id}_er.png").absolute()),
                    "b_image": str((img_dir / f"{sample_id}_nucleus.png").absolute()),
                    "g_image": str((img_dir / f"{sample_id}_protein.png").absolute()),
                    "output_prefix": sample_id,
                })
    else:
        # ---- Phase 0: Pre-compute ESM2 embeddings (streamed to HDF5) ----
        protein_sequences = load_protein_sequences(args.protein_cache)
        pups_dir = Path(args.pups_dir)
        esm2_cache_path = precompute_esm2_cache(
            npy_files, protein_sequences, esm2_cache_path, pups_dir
        )

        # ---- Phase 1: Parallel CPU preprocessing ----
        if args.skip_preprocess and preprocess_manifest_path.exists():
            print(f"\nSkipping preprocessing, loading manifest from {preprocess_manifest_path}")
            with open(preprocess_manifest_path) as f:
                preprocess_manifest = json.load(f)
            skipped = len(npy_files) - len(preprocess_manifest)
        else:
            preprocess_manifest, skipped = run_parallel_preprocessing(
                npy_files, args.raw_hpa_dir, args.perform_unmixing,
                intermediate_dir, args.num_workers,
            )
            # Save manifest
            with open(preprocess_manifest_path, "w") as f:
                json.dump(preprocess_manifest, f, indent=2)

        # Filter manifest to only FOVs with ESM2 embeddings
        with h5py.File(esm2_cache_path, "r") as f:
            cached_uids = set(f["X_esm2"].keys())
        preprocess_manifest = [
            e for e in preprocess_manifest if e["uniprot_id"] in cached_uids
        ]

        # ---- Phase 2: Batched GPU inference ----
        records_true, records_gen, all_crop_names, all_filenames_for_crops = (
            run_batched_inference(
                preprocess_manifest, esm2_cache_path, args.pups_ckpt, pups_dir,
                images_dir_true, images_dir_gen, args.crop_batch_size, device=device,
            )
        )

        print(f"\nGenerated {len(all_crop_names)} crops from {len(preprocess_manifest)} FOVs, skipped {skipped} FOVs")

        # Save manifest for caching
        manifest = {
            "crop_names": all_crop_names,
            "source_filenames": all_filenames_for_crops,
            "skipped": skipped,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    if len(all_crop_names) == 0:
        print("ERROR: No crops generated. Exiting.")
        return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PUPS on protein localization evaluation"
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="/n/netscratch/chenf2011_lab/sandeep/data/hpa_dataset/hpa_preprocessed_split_256_code_flip_ten_crop_rotate_esm_embed_mean_pool",  # change to your path
        help="CA_LlamaGen code directory (contains ca256_codes/{split}/)",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="val1",
        help="Validation split name (e.g., val1, val2, cell_line_holdouts)",
    )
    parser.add_argument(
        "--raw_hpa_dir",
        type=str,
        default="/n/netscratch/chenf2011_lab/sandeep/data/hpa_dataset/humanproteinatlas_temp",  # change to your path
        help="Path to raw HPA TIFFs (humanproteinatlas_temp/)",
    )
    parser.add_argument(
        "--pups_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "PUPS"),
        help="Path to PUPS codebase",
    )
    parser.add_argument(
        "--pups_ckpt",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "PUPS",
            "checkpoints",
            "splice_isoform_dataset_cell_line_and_gene_split_full-epoch=01-val_combined_loss=0.18.ckpt",
        ),
        help="Path to PUPS model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/n/netscratch/chenf2011_lab/sandeep/evaluation_results_benchmarks/PUPS",  # change to your path
        help="Output directory",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of FOV samples to evaluate (None = all)",
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
        "--perform_unmixing",
        action="store_true",
        default=False,
        help="Perform spectral unmixing (default: False, matching PUPS defaults)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Parallelism
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel CPU workers for preprocessing (Phase 1).",
    )
    parser.add_argument(
        "--crop_batch_size",
        type=int,
        default=64,
        help="Number of crops to batch through PUPS inference at once (Phase 2).",
    )
    parser.add_argument(
        "--skip_preprocess",
        action="store_true",
        default=False,
        help="Skip preprocessing if intermediate crops already exist.",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        default=False,
        help="Skip evaluation (default: False)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("PUPS Protein Localization Benchmark")
    print("=" * 60)
    print(f"Val dir: {args.val_dir}")
    print(f"Val split: {args.val_split}")
    print(f"Raw HPA dir: {args.raw_hpa_dir}")
    print(f"PUPS dir: {args.pups_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Num samples: {args.num_samples or 'all'}")
    print(f"Spectral unmixing: {args.perform_unmixing}")
    print(f"Num workers: {args.num_workers}")
    print(f"Crop batch size: {args.crop_batch_size}")
    print("=" * 60 + "\n")

    evaluate(args)


if __name__ == "__main__":
    main()
