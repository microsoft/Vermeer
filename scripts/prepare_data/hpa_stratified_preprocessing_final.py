#!/usr/bin/env python3
"""
## TODO: separate preprocessing and dataset split into two different scripts 
HPA Dataset Splitting

This script creates train/val1/val2/val3 splits with the following key constraints:
1. FOV 1 & 2 for same cell_line × protein combo stay together
2. Stratification by cell-line (approximate) 
3. Stratification by protein localization class for single-localizing proteins 
4. Randomly split multi-localizing proteins
5. Val1: Unique cell_line x protein combos (stratified among constitutive, deterministic, and stochastic proteins)
6. Val2: Unique proteins, controlled for sequence similarity to train proteins
7. Val3: Unique cell-lines [GAMG, SK-MEL-30, Hep-G2] (may contain unique proteins as well)

Output directories:
- train/: Training set [80%]
- val1/: Validation set 1 (unique cell_line x protein combos) [10%]
- val2/: Validation set 2 (unique proteins, controlled for sequence similarity to train) [10%]
- val3/: Validation set 3 (unique cell-lines [GAMG, SK-MEL-30, Hep-G2])
- train_sequence_similarity_removed/: Train proteins similar to val2
"""

import os
import sys
import shutil
import random
import argparse
import tempfile
import subprocess
import time
import glob
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set, Optional
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

import pandas as pd
import numpy as np
from PIL import Image
from tifffile import imread
import skimage.exposure
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import requests

import matplotlib.pyplot as plt
import ast
from Bio import SeqIO

# =============================================================================
# CONFIGURATION
# =============================================================================

config = {
    # Data paths
    'input_dir': '/n/netscratch/chenf2011_lab/sandeep/data/hpa_dataset_02272026/humanproteinatlas',  # change to your path
    'output_dir': '/n/netscratch/chenf2011_lab/sandeep/data/hpa_dataset/',  # change to your path
    'hpa_annotation_csv': '/n/home08/skambhampati/ar-microscopy-gen/code/data_preparation/hpa_cell-line-download.csv',  # change to your path
    'hpa_samples_df_csv': '/n/home08/skambhampati/ar-microscopy-gen/code/data_preparation/hpa_samples_df.csv',  # change to your path
    'train_fasta': '/n/home08/skambhampati/ar-microscopy-gen/code/data_preparation/train_proteins.fasta',  # change to your path
    'val1_fasta': '/n/home08/skambhampati/ar-microscopy-gen/code/data_preparation/val1_proteins.fasta',  # change to your path
    'output_split_distributions_png': '/n/home08/skambhampati/ar-microscopy-gen/code/data_preparation/plots/split_distributions.png',  # change to your path
    'output_split_distributions_top_15_png': '/n/home08/skambhampati/ar-microscopy-gen/code/data_preparation/plots/split_distributions_top_15.png',  # change to your path

    # Cell line holdouts
    'holdout_cell_lines': ['GAMG', 'SK-MEL-30', 'Hep-G2'],

    # Splitting parameters
    # Note: Splitting is now two-stage:
    # Stage 1: 90/10 train/val1 split (allows protein overlap)
    # Stage 2: 90/10 split of stage1 train → final train/val2 (unique proteins)
    'train_ratio_split_1': 0.9,
    'train_ratio_split_2': 0.9,
    'random_seed': 42,

    # Processing parameters
    # 'image_size': (384, 384),
    'channel_names': ['red', 'green', 'blue', 'yellow'],

    # Output formats
    'save_individual_channels': True,
    'save_concatenated_arrays': True,

    # MMSeqs2 parameters
    'run_mmseqs2': True,
    'mmseqs_threads': 8,
    'mmseqs_sensitivity': 7,
    'mmseqs_coverage': 0.8,
    'mmseqs_cov_mode': 0,
    'mmseqs_min_seq_id': 0.5,  # 50% sequence identity threshold
    'mmseqs_alignment_mode': 3,
    'mmseqs_max_seqs': 300,

    # UniProt API settings
    'uniprot_api_url': 'https://rest.uniprot.org/uniprotkb',
    'uniprot_retry_attempts': 3,
    'uniprot_retry_delay': 2,

    # Performance settings
    'parallel_fetch': True,
    'max_workers': 7,
    'use_cache': True,
}

# Set random seeds for reproducibility
random.seed(config['random_seed'])
np.random.seed(config['random_seed'])

# HPA subcellular localization classes
ALL_LOCATIONS = {
    "Nucleoplasm": 0,
    "Nuclear membrane": 1,
    "Nucleoli": 2,
    "Nucleoli fibrillar center": 3,
    "Nuclear speckles": 4,
    "Nuclear bodies": 5,
    "Endoplasmic reticulum": 6,
    "Golgi apparatus": 7,
    "Intermediate filaments": 8,
    "Actin filaments": 9,
    "Focal adhesion sites": 9,
    "Microtubules": 10,
    "Mitotic spindle": 11,
    "Centrosome": 12,
    "Centriolar satellite": 12,
    "Plasma membrane": 13,
    "Cell Junctions": 13,
    "Mitochondria": 14,
    "Aggresome": 15,
    "Cytosol": 16,
    "Vesicles": 17,
    "Peroxisomes": 17,
    "Endosomes": 17,
    "Lysosomes": 17,
    "Lipid droplets": 17,
    "Cytoplasmic bodies": 17,
    "No staining": 18
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_tiff_image(file_path: str) -> np.ndarray:
    """Load a TIFF image using PIL or tifffile."""
    try:
        with Image.open(file_path) as img:
            return np.array(img)
    except Exception as e:
        try:
            return imread(file_path)
        except Exception as e2:
            raise Exception(f"Failed to read image with both PIL and tifffile: {e}, {e2}")

def rescale_and_normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Rescale and normalize an image using percentile-based intensity scaling.
    """

    assert np.min(img) >= 0, "Image has negative values"
    # Handle edge case where image is all zeros or has no variation
    if np.max(img) == 0:
        return img
    
    p1, p99 = np.percentile(img, (1, 99))
    img_rescaled = skimage.exposure.rescale_intensity(img, in_range=(p1, p99))
    img_rescaled_and_normalized = img_rescaled / np.max(img_rescaled)

    return img_rescaled_and_normalized.astype(np.float32)

def handle_special_cell_line_name(cell_line: str) -> str:
    """
    Handle special cell line name replacements.
    """
    if "RPTEC_TERT1" in cell_line:
        return cell_line.replace("RPTEC_TERT1", "RPTEC/TERT1")
    if "HUVEC_TERT2" in cell_line:
        return cell_line.replace("HUVEC_TERT2", "HUVEC/TERT2")
    if "PODO_TERT25" in cell_line:
        return cell_line.replace("PODO_TERT25", "PODO/TERT25")
    if "PODO_TERT256" in cell_line:
        return cell_line.replace("PODO_TERT256", "PODO/TERT256")
    if "PODO_SVTERT152" in cell_line:
        return cell_line.replace("PODO_SVTERT152", "PODO/SVTERT152")

    return cell_line

def parse_directory_path(directory_path: str, base_input_dir: str) -> Dict[str, str]:
    """
    Parse HPA directory path to extract metadata.
    Expected structure: .../uniprot_id/antibody_id/cell_line/fov_folder/
    Some cell lines have slashes in their names, appearing as subdirectories.
    """
    rel_path = os.path.relpath(directory_path, base_input_dir)
    path_parts = rel_path.split(os.sep)

    # Handle cell lines with slashes (e.g., RPTEC/TERT1, PODO/SVTERT152)
    # These appear as extra directory levels
    special_cell_line_suffixes = ['TERT1', 'TERT2', 'TERT25', 'TERT256', 'SVTERT152']
    
    if len(path_parts) >= 5 and path_parts[3] in special_cell_line_suffixes:
        # Cell line is split across two directories
        uniprot = path_parts[0]
        antibody = path_parts[1]
        cell_line = f"{path_parts[2]}/{path_parts[3]}"
        fov = path_parts[4]
        original_cell_line = f"{path_parts[2]}_{path_parts[3]}"
        
    elif len(path_parts) >= 4:
        # Standard case
        uniprot = path_parts[0]
        antibody = path_parts[1]
        cell_line = path_parts[2]
        fov = path_parts[3]
        original_cell_line = cell_line
        
    else:
        raise ValueError(f"Directory path {directory_path} does not have enough parts")

    return {
        'uniprot': uniprot,
        'antibody': antibody,
        'cell_line': cell_line,
        'fov': fov,
        'full_id': f"{uniprot}_{antibody}_{original_cell_line}_{fov}",
        'directory_path': directory_path,
        'relative_path': rel_path
    }

def create_directory_structure(base_dir: str, formats: List[str], splits: List[str]):
    """Create the output directory structure."""
    for format_name in formats:
        for split in splits:
            class_dir = os.path.join(base_dir, format_name, split, "0000")
            os.makedirs(class_dir, exist_ok=True)

    os.makedirs(os.path.join(base_dir, "metadata"), exist_ok=True)

def plot_split_distributions(split_dict, plot_top_15=False):
    """
    Plot distributions of cell lines, protein types, and localizations across splits.
    
    Args:
        split_dict: Dictionary with 'train', 'val1', 'val2' DataFrames
        plot_top_15: If True, show only top 15 cell lines and localizations. If False, show all.
    """
    
    train_df = split_dict['train']
    val1_df = split_dict['val1']
    val2_df = split_dict['val2']
    
    # Set larger font sizes
    plt.rcParams.update({'font.size': 18})

    # Scale figure height when showing all categories so ytick labels are readable
    if plot_top_15:
        fig_height = 12
        height_ratios = [1.5, 1]
    else:
        n_cell_lines = len(set(train_df['cell_line']) | set(val1_df['cell_line']) | set(val2_df['cell_line']))
        fig_height = max(12, 0.4 * n_cell_lines + 8)
        height_ratios = [n_cell_lines, max(n_cell_lines * 0.7, 8)]

    # Create figure with subplots: top row (cell line), bottom row (protein type, localization)
    fig = plt.figure(figsize=(18, fig_height))
    gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.3, width_ratios=[1, 2.33], height_ratios=height_ratios)
    
    # ============ Top Row: Cell Line Distribution ============
    ax1 = fig.add_subplot(gs[0, :])  # Spans both columns
    
    # Cell line frequencies
    train_freq = train_df['cell_line'].value_counts() / len(train_df)
    val1_freq = val1_df['cell_line'].value_counts() / len(val1_df)
    val2_freq = val2_df['cell_line'].value_counts() / len(val2_df)
    
    # Align
    all_cell_lines = sorted(set(train_freq.index) | set(val1_freq.index) | set(val2_freq.index))
    train_freq = train_freq.reindex(all_cell_lines, fill_value=0)
    val1_freq = val1_freq.reindex(all_cell_lines, fill_value=0)
    val2_freq = val2_freq.reindex(all_cell_lines, fill_value=0)
    
    # Sort by train frequency
    sorted_idx = train_freq.argsort()[::-1]
    if plot_top_15:
        top_n = min(15, len(all_cell_lines))
        all_cell_lines_sorted = [all_cell_lines[i] for i in sorted_idx[:top_n]]
        train_freq_sorted = train_freq.values[sorted_idx[:top_n]]
        val1_freq_sorted = val1_freq.values[sorted_idx[:top_n]]
        val2_freq_sorted = val2_freq.values[sorted_idx[:top_n]]
    else:
        all_cell_lines_sorted = [all_cell_lines[i] for i in sorted_idx]
        train_freq_sorted = train_freq.values[sorted_idx]
        val1_freq_sorted = val1_freq.values[sorted_idx]
        val2_freq_sorted = val2_freq.values[sorted_idx]
    
    # Plot
    x = np.arange(len(all_cell_lines_sorted))
    width = 0.25
    ax1.barh(x - width, train_freq_sorted, width, label=f'Train', color='steelblue')
    ax1.barh(x, val1_freq_sorted, width, label=f'Unseen Proteins', color='coral')
    ax1.barh(x + width, val2_freq_sorted, width, label=f'Unseen Proteins x Cell-Lines', color='tab:green')

    print(f"Train: {len(train_df)}, Val1: {len(val1_df)}, Val2: {len(val2_df)}")
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(all_cell_lines_sorted, fontsize=16, rotation=30, ha='right')
    ax1.set_xlabel('Frequency', fontsize=22)
    ax1.tick_params(axis='x', labelsize=20)
    title_suffix = ' (Top 15)' if plot_top_15 else ''
    ax1.set_title(f'Cell Line Distribution{title_suffix}', fontsize=26, fontweight='bold')
    ax1.legend(fontsize=21)
    ax1.grid(axis='x', alpha=0.3)
    
    # ============ Bottom Left: Protein Type Distribution ============
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Protein type frequencies
    train_freq = train_df['protein_type'].value_counts() / len(train_df)
    val1_freq = val1_df['protein_type'].value_counts() / len(val1_df)
    val2_freq = val2_df['protein_type'].value_counts() / len(val2_df)
    
    # Align
    all_categories = sorted(set(train_freq.index) | set(val1_freq.index) | set(val2_freq.index))
    train_freq = train_freq.reindex(all_categories, fill_value=0)
    val1_freq = val1_freq.reindex(all_categories, fill_value=0)
    val2_freq = val2_freq.reindex(all_categories, fill_value=0)
    
    # Sort by train frequency
    sorted_idx = train_freq.argsort()[::-1]
    all_categories_sorted = [all_categories[i] for i in sorted_idx]
    train_freq_sorted = train_freq.values[sorted_idx]
    val1_freq_sorted = val1_freq.values[sorted_idx]
    val2_freq_sorted = val2_freq.values[sorted_idx]
    
    # Plot
    x = np.arange(len(all_categories_sorted))
    width = 0.25
    ax2.barh(x - width, train_freq_sorted, width, label=f'Train', color='steelblue')
    ax2.barh(x, val1_freq_sorted, width, label=f'Unseen Proteins', color='coral')
    ax2.barh(x + width, val2_freq_sorted, width, label=f'Unseen Proteins x Cell-Lines', color='tab:green')
    
    ax2.set_yticks(x)
    ax2.set_yticklabels(all_categories_sorted, fontsize=18)
    ax2.set_xlabel('Frequency', fontsize=22)
    ax2.tick_params(axis='x', labelsize=20)
    ax2.set_title('Protein Type Distribution', fontsize=26, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # ============ Bottom Right: Primary Localization Distribution ============
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Filter for single-localizing proteins
    train_single = train_df[~train_df['is_multi_localizing']].copy()
    val1_single = val1_df[~val1_df['is_multi_localizing']].copy()
    val2_single = val2_df[~val2_df['is_multi_localizing']].copy()
    
    if len(train_single) > 0 or len(val1_single) > 0 or len(val2_single) > 0:
        # Extract primary localization
        if len(train_single) > 0:
            train_single['primary_loc'] = train_single['localizations'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'unknown')
            train_freq = train_single['primary_loc'].value_counts() / len(train_single)
        else:
            train_freq = pd.Series()
        
        if len(val1_single) > 0:
            val1_single['primary_loc'] = val1_single['localizations'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'unknown')
            val1_freq = val1_single['primary_loc'].value_counts() / len(val1_single)
        else:
            val1_freq = pd.Series()
        
        if len(val2_single) > 0:
            val2_single['primary_loc'] = val2_single['localizations'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'unknown')
            val2_freq = val2_single['primary_loc'].value_counts() / len(val2_single)
        else:
            val2_freq = pd.Series()

        # Align
        all_locs = sorted(set(train_freq.index) | set(val1_freq.index) | set(val2_freq.index))
        train_freq = train_freq.reindex(all_locs, fill_value=0)
        val1_freq = val1_freq.reindex(all_locs, fill_value=0)
        val2_freq = val2_freq.reindex(all_locs, fill_value=0)

        # Sort by train frequency
        sorted_idx = train_freq.argsort()[::-1]
        if plot_top_15:
            top_n = min(15, len(all_locs))
            all_locs_sorted = [all_locs[i] for i in sorted_idx[:top_n]]
            train_freq_sorted = train_freq.values[sorted_idx[:top_n]]
            val1_freq_sorted = val1_freq.values[sorted_idx[:top_n]]
            val2_freq_sorted = val2_freq.values[sorted_idx[:top_n]]
            loc_fontsize = 16
        else:
            all_locs_sorted = [all_locs[i] for i in sorted_idx]
            train_freq_sorted = train_freq.values[sorted_idx]
            val1_freq_sorted = val1_freq.values[sorted_idx]
            val2_freq_sorted = val2_freq.values[sorted_idx]
            loc_fontsize = 15

        # Plot
        x = np.arange(len(all_locs_sorted))
        width = 0.25
        ax3.barh(x - width, train_freq_sorted, width, label=f'Train', color='steelblue')
        ax3.barh(x, val1_freq_sorted, width, label=f'Unseen Proteins', color='coral')
        ax3.barh(x + width, val2_freq_sorted, width, label=f'Unseen Proteins x Cell-Lines', color='tab:green')

        ax3.set_yticks(x)
        ax3.set_yticklabels(all_locs_sorted, fontsize=loc_fontsize)
        ax3.set_xlabel('Frequency', fontsize=22)
        ax3.tick_params(axis='x', labelsize=20)
        loc_suffix = ' (Top 15)' if plot_top_15 else ''
        ax3.set_title(f'Localization{loc_suffix} (Single Localizing)', fontsize=26, fontweight='bold')
        ax3.legend(fontsize=21)
        ax3.grid(axis='x', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No single-localizing\nproteins found',
                 ha='center', va='center', fontsize=21, transform=ax3.transAxes)
        ax3.set_title('Primary Localization Distribution', fontsize=26, fontweight='bold')
    
    return fig

# =============================================================================
# PHASE 1: DATA LOADING FUNCTIONS
# =============================================================================

def load_hpa_annotations(hpa_csv_path: str) -> pd.DataFrame:
    """
    Load HPA annotation CSV and optionally filter out holdout cell lines.
    """
    print(f"Loading HPA annotations from {hpa_csv_path}")

    if not os.path.exists(hpa_csv_path):
        raise FileNotFoundError(f"HPA annotation CSV not found: {hpa_csv_path}")

    hpa_df = pd.read_csv(hpa_csv_path)

    print(f"Loaded {len(hpa_df)} annotation records")
    print(f"Columns: {list(hpa_df.columns)}")

    return hpa_df

def get_localization_from_annotation(sample_info: Dict[str, Any], hpa_annot_df: pd.DataFrame) -> List[str]:
    """
    Get localization information from HPA annotation data.
    """
    if hpa_annot_df is None or hpa_annot_df.empty:
        return []

    uniprot = sample_info.get('uniprot', '')
    antibody = sample_info.get('antibody', '')
    cell_line = sample_info.get('cell_line', '')
    fov = sample_info.get('fov', '').replace('fov_', '')

    # Handle special cell line name replacements (reverse mapping for annotation lookup)
    cell_line = handle_special_cell_line_name(cell_line)

    fov_int = int(fov)

    # # Convert fov to integer
    # try:
    #     fov_int = int(fov)
    # except (ValueError, TypeError):
    #     fov_int = 1

    # Look up in annotation dataframe
    matches = hpa_annot_df[
        (hpa_annot_df['uniprot'] == uniprot) &
        (hpa_annot_df['antibody'] == antibody) &
        (hpa_annot_df['cell_line'] == cell_line) &
        (hpa_annot_df['fov'] == fov_int)
    ]['localizations']

    if len(matches) > 0:
        localization_str = matches.values[0]
        # Parse string representation of list
        localization_list = eval(localization_str)
        return localization_list
    else:
        return []

def extract_localization_info(sample_info: Dict[str, Any], hpa_annot_df: pd.DataFrame = None) -> List[str]:
    """
    Extract localization information.
    """
    localizations = get_localization_from_annotation(sample_info, hpa_annot_df)

    if len(localizations) == 0:
        localizations = ['unknown'] 

    return localizations

def discover_hpa_data(input_dir: str, hpa_annot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Discover HPA data and create sample inventory, skipping sperm cells
    """
    print("\n" + "="*60)
    print("PHASE 1: DATA DISCOVERY")
    print("="*60)
    print(f"Scanning: {input_dir}")

    samples = []
    cell_line_counts = defaultdict(int)
    error_count = 0

    for root, dirs, files in tqdm(os.walk(input_dir), desc="Scanning directories"):
        # Skip sperm directories
        if 'sperm' in root.lower():
            continue

        # Look for TIFF files
        tiff_files = [f for f in files if f.endswith('.tif') or f.endswith('.tiff')]
        if not tiff_files:
            continue

        # Find channel files
        channel_files = {'red': None, 'green': None, 'blue': None, 'yellow': None}
        for file in tiff_files:
            file_lower = file.lower()
            if 'red' in file_lower:
                channel_files['red'] = file
            elif 'green' in file_lower:
                channel_files['green'] = file
            elif 'blue' in file_lower:
                channel_files['blue'] = file
            elif 'yellow' in file_lower:
                channel_files['yellow'] = file

        # Require all 4 channels
        available_channels = [ch for ch, file in channel_files.items() if file is not None]
        if len(available_channels) == 4:
            try:
                metadata = parse_directory_path(root, input_dir)
                cell_line = metadata.get('cell_line', 'unknown')

                base_filename = metadata.get('full_id', os.path.basename(root))

                sample_info = {
                    'directory': root,
                    'base_filename': base_filename,
                    'metadata': metadata,
                    'channel_files': channel_files,
                    'available_channels': available_channels,
                    'cell_line': cell_line,
                    'uniprot': metadata.get('uniprot', 'unknown'),
                    'antibody': metadata.get('antibody', 'unknown'),
                    'fov': metadata.get('fov', 'unknown'),
                }

                localizations = extract_localization_info(sample_info, hpa_annot_df)
                sample_info['localizations'] = localizations

                samples.append(sample_info)
                cell_line_counts[cell_line] += 1

            except Exception as e:
                error_count += 1
                print(f"Error processing {root}: {e}")

    samples_df = pd.DataFrame(samples)

    print(f"\nDiscovery summary:")
    print(f"  Total samples: {len(samples_df)}")
    print(f"  Errors: {error_count}")
    print(f"  Unique proteins: {samples_df['uniprot'].nunique()}")
    print(f"  Unique cell lines: {samples_df['cell_line'].nunique()}")
    print(f"  Unique antibodies: {samples_df['antibody'].nunique()}")

    return samples_df


def create_protein_cellline_groups(samples_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group samples by protein × cell line to ensure FOV 1 & 2 stay together.
    """
    print("\n" + "="*60)
    print("CREATING PROTEIN×CELLLINE GROUPS")
    print("="*60)

    # Group by uniprot + antibody + cell_line
    grouped = samples_df.groupby(['uniprot', 'antibody', 'cell_line']).agg({
        'base_filename': list,  # All FOV samples
        'localizations': list,  # all localizations across fovs 
        'fov': 'count'  # Number of FOVs
    }).reset_index()

    grouped.rename(columns={'fov': 'num_fovs'}, inplace=True)
    grouped['protein_cellline_id'] = (
        grouped['uniprot'] + '_' + grouped['cell_line']
    )

    print(f"Created {len(grouped)} protein×cellline groups")
    print(f"  From {samples_df['uniprot'].nunique()} unique proteins")
    print(f"  Across {samples_df['cell_line'].nunique()} cell lines")

    return grouped


# =============================================================================
# PHASE 2: PROTEIN CATEGORIZATION
# =============================================================================

def categorize_protein_type(uniprot_id: str, samples_df: pd.DataFrame) -> str:
    """
    Categorize protein as constitutive, deterministic, or stochastic.

    - Constitutive: Same localization across all images (all cell lines and FOVs)
    - Deterministic: Different localizations across cell lines, but consistent within each cell line
    - Stochastic: Variable localizations between FOVs within the same cell line
    """
    protein_data = samples_df[samples_df['uniprot'] == uniprot_id].copy()
    
    if len(protein_data) == 0:
        return 'unknown'
    
    # Count unique localizations across all samples
    unique_locs = len(set(tuple(loc) for loc in protein_data['localizations']))
    
    if unique_locs == 1:
        # Same localization everywhere
        return 'constitutive'
    
    # Check if different localizations exist between FOVs within the same cell line
    # Group by cell_line and protein, then check if different FOVs have different localizations
    for (cell_line, protein_id) in protein_data[['cell_line', 'uniprot']].drop_duplicates().values:
        cell_antibody_data = protein_data[
            (protein_data['cell_line'] == cell_line) & 
            (protein_data['uniprot'] == protein_id)
        ]
        
        # Check for different localizations across FOVs for this cell line/antibody combination
        unique_locs_per_fov = len(set(tuple(loc) for loc in cell_antibody_data['localizations']))
        
        if unique_locs_per_fov > 1:
            # Found different localizations between FOVs of same cell line/antibody = stochastic
            return 'stochastic'
    
    # Different localizations exist, but only across different cell lines = deterministic
    return 'deterministic'


def is_multi_localizing(uniprot_id: str, protein_groups_df: pd.DataFrame) -> bool:
    """
    Determine if protein is multi-localizing (has multiple annotations per sample).
    """
    protein_data = protein_groups_df[protein_groups_df['uniprot'] == uniprot_id]
    # If ANY sample has len(localizations) > 1
    return any(len(loc) > 1 for loc in protein_data['localizations'])


def categorize_proteins(protein_groups_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add protein_type and is_multi_localizing columns.
    """
    print("\n" + "="*60)
    print("PHASE 2: PROTEIN CATEGORIZATION")
    print("="*60)

    # Get unique proteins
    unique_proteins = protein_groups_df['uniprot'].unique()

    print(f"Categorizing {len(unique_proteins)} unique proteins...")

    # Categorize each protein
    protein_types = {}
    multi_localizing = {}

    for uniprot in tqdm(unique_proteins, desc="Categorizing proteins"):
        protein_types[uniprot] = categorize_protein_type(uniprot, protein_groups_df)
        multi_localizing[uniprot] = is_multi_localizing(uniprot, protein_groups_df)

    # Add columns
    protein_groups_df['protein_type'] = protein_groups_df['uniprot'].map(protein_types)
    protein_groups_df['is_multi_localizing'] = protein_groups_df['uniprot'].map(multi_localizing)

    # Print distribution
    print("\nProtein type distribution:")
    type_counts = pd.Series([protein_types[p] for p in unique_proteins]).value_counts()
    for ptype, count in type_counts.items():
        print(f"  {ptype}: {count} ({count/len(unique_proteins)*100:.1f}%)")

    print("\nLocalization type distribution:")
    multi_count = sum(multi_localizing.values())
    single_count = len(unique_proteins) - multi_count
    print(f"  Single-localizing: {single_count} ({single_count/len(unique_proteins)*100:.1f}%)")
    print(f"  Multi-localizing: {multi_count} ({multi_count/len(unique_proteins)*100:.1f}%)")

    return protein_groups_df


def aggregate_protein_statistics(protein_groups_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create protein-level summary (one row per protein).
    """
    print("\n" + "="*60)
    print("AGGREGATING PROTEIN-LEVEL STATISTICS")
    print("="*60)

    protein_stats = protein_groups_df.groupby('uniprot').agg({
        'protein_type': 'first',
        'is_multi_localizing': 'first',
        'cell_line': lambda x: list(x.unique()),
        # 'protein_cellline_id': 'count',
    }).reset_index()

    protein_stats.rename(columns={'protein_cellline_id': 'num_protein_cellline_groups'}, inplace=True)

    print(f"Created protein-level statistics for {len(protein_stats)} proteins")

    return protein_stats

# =============================================================================
# PHASE 3: PROTEIN-LEVEL STRATIFIED SPLIT
# =============================================================================


def create_stratified_splits(samples_df: pd.DataFrame, 
                           train_ratio: float = 0.9,
                           random_seed: int = 42,
                           min_samples_per_cell_line: int = 10,
                           use_localization_stratification: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Create stratified train/val splits.
    
    Strategy:
    - Split by (cell_line × protein) groups to keep FOVs together
    - Stratify by cell_line + primary_localization for single_localizing proteins 
    - Stratify by cell_line + protein_type (deterministic, stochastic, constitutive) for multi_localizing proteins
    - Proteins can overlap between train and val (same protein with different cell lines)
    
    Args:
        samples_df: DataFrame with sample information including protein_type, is_multi_localizing
        train_ratio: Ratio for train split (default 0.9)
        random_seed: Random seed for reproducibility
        min_samples_per_cell_line: Minimum samples per cell line to include
        use_localization_stratification: Whether to stratify by localization
    
    Returns:
        Dictionary with 'train' and 'val' DataFrames
    """
    print("\n" + "="*60)
    print("PHASE 3: CREATING STRATIFIED SPLITS")
    print("="*60)

    np.random.seed(random_seed)
    random.seed(random_seed)
    
    if len(samples_df) == 0:
        return {'train': pd.DataFrame(), 'val': pd.DataFrame()}
    
    # Create grouping key to keep FOVs together (cell_line + antibody + uniprot)
    samples_df = samples_df.copy()
    samples_df['group_key'] = (
        samples_df['cell_line'] + '_' + 
        samples_df['uniprot']
    )
    
    # Get unique groups (each group contains all FOVs for a cell_line/protein combination)
    groups = []
    for group_key, group_data in samples_df.groupby('group_key'):
        # Extract primary localization for single-localizing proteins
        if group_data['is_multi_localizing'].iloc[0]:
            # Multi-localizing: use protein_type for stratification
            primary_loc = 'multi'
            stratify_value = group_data['protein_type'].iloc[0]
        else:
            # Single-localizing: use first localization
            locs = group_data['localizations'].iloc[0]
            if isinstance(locs, list) and len(locs) > 0:
                primary_loc = locs[0]
            else:
                primary_loc = 'unknown'
            stratify_value = primary_loc
        
        groups.append({
            'group_key': group_key,
            'cell_line': group_data['cell_line'].iloc[0],
            'uniprot': group_data['uniprot'].iloc[0],
            'protein_type': group_data['protein_type'].iloc[0],
            'is_multi_localizing': group_data['is_multi_localizing'].iloc[0],
            'primary_localization': primary_loc,
            'stratify_value': stratify_value,
            'num_fovs': len(group_data),
            'sample_indices': group_data.index.tolist()
        })
    
    groups_df = pd.DataFrame(groups)
    print(f"\nTotal groups (cell_line × protein): {len(groups_df)}")
    print(f"  Single-localizing groups: {(~groups_df['is_multi_localizing']).sum()}")
    print(f"  Multi-localizing groups: {groups_df['is_multi_localizing'].sum()}")
    
    # Create stratification key
    if use_localization_stratification:
        # For single-localizing: cell_line + primary_localization
        # For multi-localizing: cell_line + protein_type
        groups_df['stratify_key'] = (
            groups_df['cell_line'] + '_' + groups_df['stratify_value']
        )
    else:
        # Only stratify by cell_line
        groups_df['stratify_key'] = groups_df['cell_line']
    
    # Filter by cell line frequency (at the group level)
    cell_line_group_counts = groups_df['cell_line'].value_counts()
    valid_cell_lines = cell_line_group_counts[cell_line_group_counts >= min_samples_per_cell_line].index
    groups_df_filtered = groups_df[groups_df['cell_line'].isin(valid_cell_lines)].copy()
    
    print(f"\nCell lines after filtering (>= {min_samples_per_cell_line} groups):")
    print(f"  Valid cell lines: {len(valid_cell_lines)}")
    print(f"  Groups remaining: {len(groups_df_filtered)}")
    
    if len(groups_df_filtered) == 0:
        print("WARNING: No groups remaining after filtering!")
        return {'train': pd.DataFrame(), 'val': pd.DataFrame()}
    
    # Check stratification keys - need at least 2 samples per key for 2-way splitting
    stratify_counts = groups_df_filtered['stratify_key'].value_counts()
    valid_stratify_keys = stratify_counts[stratify_counts >= 2].index
    
    groups_valid = groups_df_filtered[groups_df_filtered['stratify_key'].isin(valid_stratify_keys)].copy()
    groups_rare = groups_df_filtered[~groups_df_filtered['stratify_key'].isin(valid_stratify_keys)].copy()
    
    print(f"\nStratification keys:")
    print(f"  Valid keys (>= 2 groups): {len(valid_stratify_keys)}")
    print(f"  Groups with valid keys: {len(groups_valid)}")
    print(f"  Groups with rare keys: {len(groups_rare)}")
    
    # Split groups into train and val
    train_groups_list = []
    val_groups_list = []

    # Split groups with valid stratification keys
    if len(groups_valid) > 0:
        train_idx, val_idx = train_test_split(
            range(len(groups_valid)),
            test_size=(1 - train_ratio),
            random_state=random_seed,
            stratify=groups_valid['stratify_key']
        )
        train_groups_list.append(groups_valid.iloc[train_idx])
        val_groups_list.append(groups_valid.iloc[val_idx])

    # Randomly split groups with rare keys
    if len(groups_rare) > 0:
        if len(groups_rare) >= 2:
            train_idx_rare, val_idx_rare = train_test_split(
                range(len(groups_rare)),
                test_size=(1 - train_ratio),
                random_state=random_seed
            )
            train_groups_list.append(groups_rare.iloc[train_idx_rare])
            val_groups_list.append(groups_rare.iloc[val_idx_rare])
        else:
            # Only 1 rare group, assign to train
            train_groups_list.append(groups_rare)
    
    # Combine groups
    train_groups = pd.concat(train_groups_list) if train_groups_list else pd.DataFrame()
    val_groups = pd.concat(val_groups_list) if val_groups_list else pd.DataFrame()
    
    # Extract sample indices from groups
    train_indices = []
    for indices_list in train_groups['sample_indices']:
        train_indices.extend(indices_list)
    
    val_indices = []
    for indices_list in val_groups['sample_indices']:
        val_indices.extend(indices_list)
    
    # Create DataFrames
    train_samples = samples_df.loc[train_indices].copy() if train_indices else pd.DataFrame()
    val_samples = samples_df.loc[val_indices].copy() if val_indices else pd.DataFrame()
    
    # Remove the temporary group_key column
    if len(train_samples) > 0:
        train_samples = train_samples.drop(columns=['group_key'])
    if len(val_samples) > 0:
        val_samples = val_samples.drop(columns=['group_key'])
    
    # Print statistics
    print(f"\n{'='*60}")
    print("SPLIT RESULTS")
    print(f"{'='*60}")
    print(f"Train:")
    print(f"  Groups: {len(train_groups)}")
    print(f"  Samples: {len(train_samples)}")
    print(f"  Unique proteins: {train_samples['uniprot'].nunique() if len(train_samples) > 0 else 0}")
    print(f"  Unique cell lines: {train_samples['cell_line'].nunique() if len(train_samples) > 0 else 0}")
    
    print(f"\nValidation:")
    print(f"  Groups: {len(val_groups)}")
    print(f"  Samples: {len(val_samples)}")
    print(f"  Unique proteins: {val_samples['uniprot'].nunique() if len(val_samples) > 0 else 0}")
    print(f"  Unique cell lines: {val_samples['cell_line'].nunique() if len(val_samples) > 0 else 0}")
    
    # Check for overlaps
    if len(train_samples) > 0 and len(val_samples) > 0:
        train_proteins = set(train_samples['uniprot'])
        val_proteins = set(val_samples['uniprot'])
        
        overlap_train_val = train_proteins & val_proteins
        
        print(f"\nProtein overlap:")
        print(f"  Train ∩ Val: {len(overlap_train_val)}")
        print(f"  Note: Overlaps are expected for cell_line × protein splits")
    
    # Show distribution by protein type
    if len(train_samples) > 0:
        print(f"\nTrain protein type distribution:")
        for ptype, count in train_samples['protein_type'].value_counts().items():
            print(f"  {ptype}: {count} ({count/len(train_samples)*100:.1f}%)")
    
    if len(val_samples) > 0:
        print(f"\nVal protein type distribution:")
        for ptype, count in val_samples['protein_type'].value_counts().items():
            print(f"  {ptype}: {count} ({count/len(val_samples)*100:.1f}%)")
    
    return {
        'train': train_samples,
        'val': val_samples
    }


def create_unique_protein_split(samples_df: pd.DataFrame,
                                train_ratio: float = 0.9,
                                random_seed: int = 42,
                                use_localization_stratification: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Split by PROTEINS (not cell_line × protein groups) to ensure unique proteins in val.
    
    Strategy:
    - Group all samples by protein (uniprot) only
    - Ignore cell_line information - stratify only by protein characteristics
    - For single-localizing: stratify by primary localization
    - For multi-localizing: stratify by protein_type (deterministic/constitutive/stochastic)
    - All samples of a protein (across all cell lines) assigned to same split
    
    Args:
        samples_df: DataFrame with sample information including protein_type, is_multi_localizing
        train_ratio: Ratio for train split (default 0.9)
        random_seed: Random seed for reproducibility
        use_localization_stratification: Whether to stratify by localization
    
    Returns:
        Dictionary with 'train' and 'val' DataFrames containing completely disjoint proteins
    """
    print("\n" + "="*60)
    print("CREATING UNIQUE PROTEIN SPLIT")
    print("="*60)
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    if len(samples_df) == 0:
        return {'train': pd.DataFrame(), 'val': pd.DataFrame()}
    
    samples_df = samples_df.copy()
    
    # Group samples by protein (uniprot) - ignore cell_line
    protein_groups = []
    for uniprot, protein_data in samples_df.groupby('uniprot'):
        # Determine if protein is multi-localizing
        is_multi_localizing = protein_data['is_multi_localizing'].iloc[0]
        protein_type = protein_data['protein_type'].iloc[0]
        
        if is_multi_localizing:
            # Multi-localizing: use protein_type for stratification
            stratify_value = protein_type
        else:
            # Single-localizing: find most common localization across all samples
            all_locs = []
            for locs in protein_data['localizations']:
                if isinstance(locs, list) and len(locs) > 0:
                    all_locs.append(locs[0])
            
            if all_locs:
                # Use most common localization
                from collections import Counter
                stratify_value = Counter(all_locs).most_common(1)[0][0]
            else:
                stratify_value = 'unknown'
        
        protein_groups.append({
            'uniprot': uniprot,
            'protein_type': protein_type,
            'is_multi_localizing': is_multi_localizing,
            'stratify_value': stratify_value,
            'num_samples': len(protein_data),
            'sample_indices': protein_data.index.tolist()
        })
    
    proteins_df = pd.DataFrame(protein_groups)
    print(f"\nTotal unique proteins: {len(proteins_df)}")
    print(f"  Single-localizing proteins: {(~proteins_df['is_multi_localizing']).sum()}")
    print(f"  Multi-localizing proteins: {proteins_df['is_multi_localizing'].sum()}")
    
    # Create stratification key (only by protein characteristics, no cell_line)
    if use_localization_stratification:
        proteins_df['stratify_key'] = proteins_df['stratify_value']
    else:
        # No stratification, just use a constant key
        proteins_df['stratify_key'] = 'all'
    
    # Check stratification keys - need at least 2 samples per key
    stratify_counts = proteins_df['stratify_key'].value_counts()
    valid_stratify_keys = stratify_counts[stratify_counts >= 2].index
    
    proteins_valid = proteins_df[proteins_df['stratify_key'].isin(valid_stratify_keys)].copy()
    proteins_rare = proteins_df[~proteins_df['stratify_key'].isin(valid_stratify_keys)].copy()
    
    print(f"\nStratification keys:")
    print(f"  Valid keys (>= 2 proteins): {len(valid_stratify_keys)}")
    print(f"  Proteins with valid keys: {len(proteins_valid)}")
    print(f"  Proteins with rare keys: {len(proteins_rare)}")
    
    # Split proteins into train and val
    train_proteins_list = []
    val_proteins_list = []
    
    # Split proteins with valid stratification keys
    if len(proteins_valid) > 0:
        train_idx, val_idx = train_test_split(
            range(len(proteins_valid)),
            test_size=(1 - train_ratio),
            random_state=random_seed,
            stratify=proteins_valid['stratify_key']
        )
        train_proteins_list.append(proteins_valid.iloc[train_idx])
        val_proteins_list.append(proteins_valid.iloc[val_idx])
    
    # Randomly split proteins with rare keys
    if len(proteins_rare) > 0:
        if len(proteins_rare) >= 2:
            train_idx_rare, val_idx_rare = train_test_split(
                range(len(proteins_rare)),
                test_size=(1 - train_ratio),
                random_state=random_seed
            )
            train_proteins_list.append(proteins_rare.iloc[train_idx_rare])
            val_proteins_list.append(proteins_rare.iloc[val_idx_rare])
        else:
            # Only 1 rare protein, assign to train
            train_proteins_list.append(proteins_rare)
    
    # Combine protein lists
    train_proteins = pd.concat(train_proteins_list) if train_proteins_list else pd.DataFrame()
    val_proteins = pd.concat(val_proteins_list) if val_proteins_list else pd.DataFrame()
    
    # Extract sample indices for each protein and create final sample DataFrames
    train_indices = []
    for indices_list in train_proteins['sample_indices']:
        train_indices.extend(indices_list)
    
    val_indices = []
    for indices_list in val_proteins['sample_indices']:
        val_indices.extend(indices_list)
    
    # Create final DataFrames
    train_samples = samples_df.loc[train_indices].copy() if train_indices else pd.DataFrame()
    val_samples = samples_df.loc[val_indices].copy() if val_indices else pd.DataFrame()
    
    # Print statistics
    print(f"\n{'='*60}")
    print("SPLIT RESULTS")
    print(f"{'='*60}")
    print(f"Train:")
    print(f"  Proteins: {len(train_proteins)}")
    print(f"  Samples: {len(train_samples)}")
    print(f"  Unique cell lines: {train_samples['cell_line'].nunique() if len(train_samples) > 0 else 0}")
    
    print(f"\nValidation:")
    print(f"  Proteins: {len(val_proteins)}")
    print(f"  Samples: {len(val_samples)}")
    print(f"  Unique cell lines: {val_samples['cell_line'].nunique() if len(val_samples) > 0 else 0}")
    
    # Verify no protein overlap
    if len(train_samples) > 0 and len(val_samples) > 0:
        train_protein_set = set(train_samples['uniprot'])
        val_protein_set = set(val_samples['uniprot'])
        
        overlap = train_protein_set & val_protein_set
        
        print(f"\nProtein overlap:")
        print(f"  Train ∩ Val: {len(overlap)}")
        if len(overlap) > 0:
            print(f"  WARNING: Unexpected protein overlap detected!")
        else:
            print(f"  ✓ No overlap - proteins are completely disjoint")
    
    # Show distribution by protein type
    if len(train_samples) > 0:
        print(f"\nTrain protein type distribution:")
        for ptype, count in train_samples['protein_type'].value_counts().items():
            print(f"  {ptype}: {count} ({count/len(train_samples)*100:.1f}%)")
    
    if len(val_samples) > 0:
        print(f"\nVal protein type distribution:")
        for ptype, count in val_samples['protein_type'].value_counts().items():
            print(f"  {ptype}: {count} ({count/len(val_samples)*100:.1f}%)")
    
    return {
        'train': train_samples,
        'val': val_samples
    }


# =============================================================================
# PHASE 4: MMSEQS2 CLEANUP
# =============================================================================

def fetch_protein_sequence(uniprot_id: str, api_url: str, retry_attempts: int = 3,
                           retry_delay: int = 2) -> Optional[str]:
    """
    Fetch protein sequence from UniProt API.
    """
    url = f"{api_url}/{uniprot_id}.fasta"

    for attempt in range(retry_attempts):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 404:
                return None
        except Exception as e:
            pass

        if attempt < retry_attempts - 1:
            time.sleep(retry_delay)

    return None


def fetch_and_cache_protein_sequences(unique_proteins: List[str], output_fasta: str, config: Dict) -> Set[str]:
    """
    Fetch protein sequences with caching support.
    
    Args:
        unique_proteins: List of unique UniProt IDs
        output_fasta: Output FASTA file path
        config: Configuration dictionary
        
    Returns:
        Set of successfully fetched protein IDs
    """
    cache_dir = os.path.join(os.path.dirname(output_fasta), '.protein_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'protein_sequences.pkl')

    # Load cache if exists
    cached_sequences = {}
    if os.path.exists(cache_file) and config.get('use_cache', False):
        print(f"Loading cached sequences from {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_sequences = pickle.load(f)
            print(f"Loaded {len(cached_sequences)} cached sequences")
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
            cached_sequences = {}

    proteins_to_fetch = [p for p in unique_proteins if p not in cached_sequences]

    print(f"Need to fetch {len(proteins_to_fetch)}/{len(unique_proteins)} sequences")

    fetched_proteins = set(cached_sequences.keys())

    # Fetch missing sequences
    if proteins_to_fetch:
        if config.get('parallel_fetch', False):
            # Parallel fetching
            rate_limiter = Semaphore(config.get('max_workers', 7))

            def fetch_with_rate_limit(uniprot_id):
                with rate_limiter:
                    sequence = fetch_protein_sequence(
                        uniprot_id,
                        config['uniprot_api_url'],
                        config['uniprot_retry_attempts'],
                        config['uniprot_retry_delay']
                    )
                    time.sleep(0.1)
                    return uniprot_id, sequence

            sequences = {}
            with ThreadPoolExecutor(max_workers=config.get('max_workers', 7)) as executor:
                futures = {executor.submit(fetch_with_rate_limit, uid): uid for uid in proteins_to_fetch}

                for future in tqdm(as_completed(futures), total=len(proteins_to_fetch), desc="Fetching sequences"):
                    uniprot_id, sequence = future.result()
                    if sequence:
                        sequences[uniprot_id] = sequence
                        cached_sequences[uniprot_id] = sequence
                        fetched_proteins.add(uniprot_id)
        else:
            # Sequential fetching
            for uniprot_id in tqdm(proteins_to_fetch, desc="Fetching sequences"):
                sequence = fetch_protein_sequence(
                    uniprot_id,
                    config['uniprot_api_url'],
                    config['uniprot_retry_attempts'],
                    config['uniprot_retry_delay']
                )

                if sequence:
                    cached_sequences[uniprot_id] = sequence
                    fetched_proteins.add(uniprot_id)

                time.sleep(0.1)

        # Update cache
        if config.get('use_cache', False):
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(cached_sequences, f)
                print(f"Updated cache with {len(proteins_to_fetch)} new sequences")
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")

    # Write FASTA file
    with open(output_fasta, 'w') as fasta_file:
        for uniprot_id in unique_proteins:
            if uniprot_id in cached_sequences:
                fasta_file.write(cached_sequences[uniprot_id])
                if not cached_sequences[uniprot_id].endswith('\n'):
                    fasta_file.write('\n')

    print(f"Successfully wrote {len(fetched_proteins)}/{len(unique_proteins)} sequences")

    return fetched_proteins


def run_mmseqs2_analysis(train_fasta: str, val_fasta: str, work_dir: str, config: Dict) -> str:
    """
    Run MMseqs2 sequence similarity analysis.
    (From hpa_stratified_preprocessing_pt2.py)
    """
    print("\nRunning MMSeqs2 analysis...")

    train_db = os.path.join(work_dir, 'trainDB')
    val_db = os.path.join(work_dir, 'valDB')
    result_db = os.path.join(work_dir, 'resultDB')
    tmp_dir = os.path.join(work_dir, 'tmp')
    output_tsv = os.path.join(work_dir, 'alignments.tsv')

    os.makedirs(tmp_dir, exist_ok=True)

    # Create databases
    print("Creating MMseqs2 databases...")
    subprocess.run(['mmseqs', 'createdb', train_fasta, train_db], check=True)
    subprocess.run(['mmseqs', 'createdb', val_fasta, val_db], check=True)

    # Search: train queries val
    print("Searching for sequence similarities (train -> val)...")
    cmd = [
        'mmseqs', 'search',
        train_db, val_db, result_db, tmp_dir,
        '--threads', str(config['mmseqs_threads']),
        '-s', str(config['mmseqs_sensitivity']),
        '-c', str(config['mmseqs_coverage']),
        '--cov-mode', str(config['mmseqs_cov_mode']),
        '--min-seq-id', str(config['mmseqs_min_seq_id']),
        '--alignment-mode', str(config['mmseqs_alignment_mode']),
        '--max-seqs', str(config['mmseqs_max_seqs'])
    ]
    subprocess.run(cmd, check=True)

    # Convert to TSV
    print("Converting results to TSV...")
    subprocess.run([
        'mmseqs', 'convertalis',
        train_db, val_db, result_db, output_tsv,
        '--format-output', 'query,target,pident,alnlen,qlen,tlen'
    ], check=True)

    return output_tsv


def parse_mmseqs2_results(tsv_file: str, min_seq_id: float = 0.5) -> Set[str]:
    """
    Parse MMseqs2 results and identify train proteins with high similarity to val.
    
    Args:
        tsv_file: Path to MMseqs2 TSV output
        min_seq_id: Minimum sequence identity threshold (0-1)
        
    Returns:
        Set of train protein IDs to remove
    """
    print(f"\nParsing MMSeqs2 results from {tsv_file}")

    if not os.path.exists(tsv_file) or os.path.getsize(tsv_file) == 0:
        print("No similarity results found")
        return set()

    # Read TSV
    results_df = pd.read_csv(tsv_file, sep='\t', names=['query', 'target', 'pident', 'alnlen', 'qlen', 'tlen'])

    print(f"Total alignments: {len(results_df)}")

    # Filter by sequence identity threshold
    high_sim = results_df[results_df['pident'] >= min_seq_id * 100]  # pident is percentage
    print(f"Alignments with ≥{min_seq_id*100}% identity: {len(high_sim)}")

    # Get unique train proteins
    proteins_to_remove = set(high_sim['query'].unique())

    print(f"Unique train proteins with ≥{min_seq_id*100}% similarity to val: {len(proteins_to_remove)}")

    return proteins_to_remove


def mmseqs2_cleanup_pipeline(split_dict: Dict[str, pd.DataFrame],
                             config: Dict) -> Dict[str, pd.DataFrame]:
    """
    Remove train proteins with ≥50% similarity to val proteins using MMSeqs2.
    
    This function runs BEFORE files are saved. It modifies the split dictionary in place,
    creating a new 'train_sequence_similarity_removed' split for proteins that are too
    similar to val1 proteins.
    
    Args:
        split_dict: Dictionary containing 'train', 'val' DataFrames
        config: Configuration dictionary with MMseqs2 parameters
        
    Returns:
        Updated split_dict with 'train_sequence_similarity_removed' split added
    """
    print("\n" + "="*60)
    print("PHASE 4: MMSEQS2 SEQUENCE SIMILARITY CLEANUP")
    print("="*60)
    print("Analyzing train vs val1 protein sequence similarity")

    train_df = split_dict['train']
    val_df = split_dict['val']

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix='hpa_mmseqs_')

    try:
        print("Fetching train protein sequences...")
        train_proteins = list(train_df['uniprot'].unique())
        # Fetch sequences for train proteins
        fetch_and_cache_protein_sequences(train_proteins, config['train_fasta'], config)

        print("Fetching val1 protein sequences...")
        val_proteins = list(val_df['uniprot'].unique())
        # Fetch sequences for val2 proteins
        fetch_and_cache_protein_sequences(val_proteins, config['val1_fasta'], config)

        print("Running MMSeqs2 analysis...")
        output_tsv_file = run_mmseqs2_analysis(config['train_fasta'], config['val1_fasta'], temp_dir, config)
        print(output_tsv_file)

        print("Parsing MMSeqs2 results...")
        # Parse results to get proteins to remove
        proteins_to_remove = parse_mmseqs2_results(output_tsv_file, config['mmseqs_min_seq_id'])

        print(f"\nFound {len(proteins_to_remove)} train proteins with ≥{config['mmseqs_min_seq_id']*100}% similarity to val1")

        # Split train into kept and removed samples
        mask = train_df['uniprot'].isin(proteins_to_remove)
        removed_samples = train_df[mask].copy()
        updated_train_samples = train_df[~mask].copy()

        # Update split_dict
        split_dict['train'] = updated_train_samples
        split_dict['train_sequence_similarity_removed'] = removed_samples

        print(f"\nUpdated splits:")
        print(f"  Train samples (kept): {len(updated_train_samples)}")
        print(f"    From {updated_train_samples['uniprot'].nunique()} unique proteins")
        print(f"  Train samples (removed due to similarity): {len(removed_samples)}")
        print(f"    From {removed_samples['uniprot'].nunique()} unique proteins")

        return split_dict

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

# =============================================================================
# PHASE 5: Save files
# =============================================================================
def process_single_sample(sample_info: Dict[str, Any],
                          output_dir: str,
                          split: str,
                          config: Dict) -> Dict[str, Any]:
    """
    Process a single sample: load, resize, normalize, and save in both formats.
    (Adapted from hpa_stratified_preprocessing.py)
    """
    try:
        directory = sample_info['directory']
        base_filename = sample_info['base_filename']
        channel_files = sample_info['channel_files']
        available_channels = sample_info['available_channels']

        # Load and process each channel
        processed_channels = {}
        for channel_name in config['channel_names']:
            if channel_name in available_channels and channel_files[channel_name] is not None:
                channel_path = os.path.join(directory, channel_files[channel_name])
                channel_img = load_tiff_image(channel_path)

                # Resize
                from PIL import Image
                channel_img_pil = Image.fromarray(channel_img)
                channel_img_resized = channel_img_pil.resize(config['image_size'], Image.Resampling.LANCZOS)
                channel_img_np = np.array(channel_img_resized) # uint16 

                # Normalize
                channel_normalized = rescale_and_normalize_image(channel_img_np) #float32, 0-1 
                processed_channels[channel_name] = channel_normalized
            else:
                raise ValueError(f"Channel {channel_name} is not available in the sample")
                # # Create black channel if missing
                # black_channel = np.zeros(config['image_size'], dtype=np.float32)
                # processed_channels.append((channel_name, black_channel))

        # Save individual channels
        if config['save_individual_channels']:
            for channel_name, channel_data in processed_channels.items():
                # Convert to 8-bit and make 3-channel grayscale
                channel_8bit = (channel_data * 255).astype(np.uint8)
                channel_rgb = np.repeat(channel_8bit[:, :, np.newaxis], 3, axis=2)

                output_path = os.path.join(
                    output_dir, 'individual_channels', split, '0000',
                    f"{base_filename}_{channel_name}.png"
                )
                Image.fromarray(channel_rgb).save(output_path)

        # Save concatenated arrays
        if config['save_concatenated_arrays']:
            channel_order = ['blue', 'red', 'yellow', 'green']
            # Stack channels: (H, W, 3, num_channels)
            channel_arrays = []
            for channel_name in channel_order:
                if channel_name in processed_channels:
                    channel_data = processed_channels[channel_name]
                    # Make 3-channel RGB grayscale
                    channel_3d = np.repeat(channel_data[:, :, np.newaxis], 3, axis=2)
                    channel_arrays.append(channel_3d)
                else:
                    raise ValueError(f"Channel {channel_name} is not available in the sample")

            concatenated = np.stack(channel_arrays, axis=-1)

            output_path = os.path.join(
                output_dir, 'concatenated_arrays', split, '0000',
                f"{base_filename}.npy"
            )
            np.save(output_path, concatenated)

        return {'success': True, 'base_filename': base_filename}

    except Exception as e:
        return {'success': False, 'base_filename': sample_info.get('base_filename', 'unknown'), 'error': str(e)}


def process_dataset_splits(train_samples_df: pd.DataFrame,
                           val1_samples_df: pd.DataFrame,
                           val2_samples_df: pd.DataFrame,
                           cell_line_holdouts_samples_df: pd.DataFrame,
                        #    train_sequence_similarity_removed_samples_df: pd.DataFrame,
                           samples_discovery_df: pd.DataFrame,
                           output_dir: str,
                           config: Dict) -> Dict:
    """
    Process and save images in both formats for train and val splits.
    """
    print("\n" + "="*60)
    print("PROCESSING AND SAVING IMAGES")
    print("="*60)

    # Create directory structure
    create_directory_structure(
        output_dir,
        ['individual_channels', 'concatenated_arrays'],
        ['train', 'val1', 'val2', 'cell_line_holdouts'] #, 'train_sequence_similarity_removed']
    )

    processing_stats = {}

    for split_name, split_df in [('train', train_samples_df), ('val1', val1_samples_df), ('val2', val2_samples_df), ('cell_line_holdouts', cell_line_holdouts_samples_df)]: #, train_sequence_similarity_removed_samples_df)]:
        print(f"\nProcessing {split_name} split: {len(split_df)} samples")

        split_stats = {'successful': 0, 'errors': 0}

        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}"):
            sample_info = row.to_dict()
            result = process_single_sample(sample_info, output_dir, split_name, config)

            if result['success']:
                split_stats['successful'] += 1
            else:
                split_stats['errors'] += 1
                if split_stats['errors'] <= 5:
                    print(f"  Error: {sample_info['base_filename']}: {result['error']}")

        processing_stats[split_name] = split_stats
        print(f"  {split_name}: {split_stats['successful']}/{len(split_df)} successful")

    return processing_stats


# =============================================================================
# METADATA GENERATION
# =============================================================================

def generate_metadata(train_samples_df: pd.DataFrame,
                      val1_samples_df: pd.DataFrame,
                      val2_samples_df: pd.DataFrame,
                      cell_line_holdouts_samples_df: pd.DataFrame,
                    #   train_sequence_similarity_removed_samples_df: pd.DataFrame,
                      output_dir: str,
                      config: Dict):
    """
    Generate comprehensive metadata files.
    """
    print("\n" + "="*60)
    print("GENERATING METADATA")
    print("="*60)

    metadata_dir = os.path.join(output_dir, 'metadata')

    # Extract fov from base_filename
    def extract_fov(df):
        if len(df) > 0:
            df['fov'] = df['base_filename'].str.extract(r'_(fov_\d+)')[0]
        return df

    train_samples_df = extract_fov(train_samples_df)
    val1_samples_df = extract_fov(val1_samples_df)
    val2_samples_df = extract_fov(val2_samples_df)
    cell_line_holdouts_samples_df = extract_fov(cell_line_holdouts_samples_df)
    # train_sequence_similarity_removed_samples_df = extract_fov(train_sequence_similarity_removed_samples_df)

    # Sample CSVs
    sample_columns = ['base_filename', 'uniprot', 'antibody', 'cell_line', 'fov',
                      'localizations', 'protein_type', 'is_multi_localizing']

    train_samples_df[sample_columns].to_csv(os.path.join(metadata_dir, 'train_samples.csv'), index=False)
    val1_samples_df[sample_columns].to_csv(os.path.join(metadata_dir, 'val1_samples.csv'), index=False)
    val2_samples_df[sample_columns].to_csv(os.path.join(metadata_dir, 'val2_samples.csv'), index=False)
    cell_line_holdouts_samples_df[sample_columns].to_csv(os.path.join(metadata_dir, 'cell_line_holdouts_samples.csv'), index=False)
    # train_sequence_similarity_removed_samples_df[sample_columns].to_csv(os.path.join(metadata_dir, 'train_sequence_similarity_removed_samples.csv'), index=False)

    # JSON versions
    train_samples_df[sample_columns].to_json(os.path.join(metadata_dir, 'train_samples.json'), orient='records', indent=2)
    val1_samples_df[sample_columns].to_json(os.path.join(metadata_dir, 'val1_samples.json'), orient='records', indent=2)
    val2_samples_df[sample_columns].to_json(os.path.join(metadata_dir, 'val2_samples.json'), orient='records', indent=2)
    cell_line_holdouts_samples_df[sample_columns].to_json(os.path.join(metadata_dir, 'cell_line_holdouts_samples.json'), orient='records', indent=2)
    # train_sequence_similarity_removed_samples_df[sample_columns].to_json(os.path.join(metadata_dir, 'train_sequence_similarity_removed_samples.json'), orient='records', indent=2)

    # Protein assignments
    protein_assignments = {
        'train': sorted(list(train_samples_df['uniprot'].unique())),
        'val1': sorted(list(val1_samples_df['uniprot'].unique())),
        'val2': sorted(list(val2_samples_df['uniprot'].unique())),
        'cell_line_holdouts': sorted(list(cell_line_holdouts_samples_df['uniprot'].unique())),
        # 'train_sequence_similarity_removed': sorted(list(train_sequence_similarity_removed_samples_df['uniprot'].unique())),
    }
    with open(os.path.join(metadata_dir, 'protein_assignments.json'), 'w') as f:
        json.dump(protein_assignments, f, indent=2)

    # Split statistics
    def get_distribution_dict(df: pd.DataFrame, column: str) -> Dict:
        return df[column].value_counts().to_dict() if len(df) > 0 else {}

    split_statistics = {
        'total_proteins': len(train_samples_df['uniprot'].unique()) + len(val1_samples_df['uniprot'].unique()) + len(val2_samples_df['uniprot'].unique()) + len(cell_line_holdouts_samples_df['uniprot'].unique()), #+ len(train_sequence_similarity_removed_samples_df['uniprot'].unique()),
        'train_proteins': len(train_samples_df['uniprot'].unique()),
        'val1_proteins': len(val1_samples_df['uniprot'].unique()),
        'val2_proteins': len(val2_samples_df['uniprot'].unique()),
        'cell_line_holdouts_proteins': len(cell_line_holdouts_samples_df['uniprot'].unique()),
        # 'train_sequence_similarity_removed_proteins': len(train_sequence_similarity_removed_samples_df['uniprot'].unique()),
        'total_samples': len(train_samples_df) + len(val1_samples_df) + len(val2_samples_df) + len(cell_line_holdouts_samples_df), #+ len(train_sequence_similarity_removed_samples_df),
        'train_samples': len(train_samples_df),
        'val1_samples': len(val1_samples_df),
        'val2_samples': len(val2_samples_df),
        'cell_line_holdouts_samples': len(cell_line_holdouts_samples_df),
        # 'train_sequence_similarity_removed_samples': len(train_sequence_similarity_removed_samples_df),

        'protein_type_distribution': {
            'train': get_distribution_dict(train_samples_df, 'protein_type'),
            'val1': get_distribution_dict(val1_samples_df, 'protein_type'),
            'val2': get_distribution_dict(val2_samples_df, 'protein_type'),
            'cell_line_holdouts': get_distribution_dict(cell_line_holdouts_samples_df, 'protein_type'),
            # 'train_sequence_similarity_removed': get_distribution_dict(train_sequence_similarity_removed_samples_df, 'protein_type'),
        },

        'localization_type_distribution': {
            'train': {
                'single': len(train_samples_df[~train_samples_df['is_multi_localizing']]) if len(train_samples_df) > 0 else 0,
                'multi': len(train_samples_df[train_samples_df['is_multi_localizing']]) if len(train_samples_df) > 0 else 0
            },
            'val1': {
                'single': len(val1_samples_df[~val1_samples_df['is_multi_localizing']]) if len(val1_samples_df) > 0 else 0,
                'multi': len(val1_samples_df[val1_samples_df['is_multi_localizing']]) if len(val1_samples_df) > 0 else 0
            },
            'val2': {
                'single': len(val2_samples_df[~val2_samples_df['is_multi_localizing']]) if len(val2_samples_df) > 0 else 0,
                'multi': len(val2_samples_df[val2_samples_df['is_multi_localizing']]) if len(val2_samples_df) > 0 else 0
            },
            'cell_line_holdouts': {
                'single': len(cell_line_holdouts_samples_df[~cell_line_holdouts_samples_df['is_multi_localizing']]) if len(cell_line_holdouts_samples_df) > 0 else 0,
                'multi': len(cell_line_holdouts_samples_df[cell_line_holdouts_samples_df['is_multi_localizing']]) if len(cell_line_holdouts_samples_df) > 0 else 0
            },
            # 'train_sequence_similarity_removed': {
            #     'single': len(train_sequence_similarity_removed_samples_df[~train_sequence_similarity_removed_samples_df['is_multi_localizing']]) if len(train_sequence_similarity_removed_samples_df) > 0 else 0,
            #     'multi': len(train_sequence_similarity_removed_samples_df[train_sequence_similarity_removed_samples_df['is_multi_localizing']]) if len(train_sequence_similarity_removed_samples_df) > 0 else 0
            # }
        },

        'cell_line_distribution': {
            'train': get_distribution_dict(train_samples_df, 'cell_line'),
            'val1': get_distribution_dict(val1_samples_df, 'cell_line'),
            'val2': get_distribution_dict(val2_samples_df, 'cell_line'),
            'cell_line_holdouts': get_distribution_dict(cell_line_holdouts_samples_df, 'cell_line'),
            # 'train_sequence_similarity_removed': get_distribution_dict(train_sequence_similarity_removed_samples_df, 'cell_line')
        },
    }

    with open(os.path.join(metadata_dir, 'split_statistics.json'), 'w') as f:
        json.dump(split_statistics, f, indent=2)

    # Config snapshot
    with open(os.path.join(metadata_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Metadata saved to {metadata_dir}")



# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main execution pipeline."""

    print("="*60)
    print("HPA DATASET SPLITTING")
    print("="*60)
    print(f"Input: {config['input_dir']}")
    print(f"Output: {config['output_dir']}")
    print(f"Holdout cell lines: {config['holdout_cell_lines']}")
    print(f"Train ratio1: {config['train_ratio_split_1']}")
    print(f"Train ratio2: {config['train_ratio_split_2']}")
    print(f"Random seed: {config['random_seed']}")
    print(f"Image size: {config['image_size']}")

    # Set random seed
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])

    #########################################################################################
    # PHASE 1: Data Discovery and Grouping
    #########################################################################################
    hpa_annot_df = load_hpa_annotations(config['hpa_annotation_csv'])

    if not os.path.exists(config['hpa_samples_df_csv']):
        samples_df = discover_hpa_data(config['input_dir'], hpa_annot_df)
        samples_df.to_csv(config['hpa_samples_df_csv'], index=False)
    else:
        samples_df = pd.read_csv(config['hpa_samples_df_csv'])
        # Parse the localizations column from string to list
        samples_df['localizations'] = samples_df['localizations'].apply(ast.literal_eval)
        # Parse the channel_files and available_channels columns from string to dict/list
        samples_df['channel_files'] = samples_df['channel_files'].apply(ast.literal_eval)
        samples_df['available_channels'] = samples_df['available_channels'].apply(ast.literal_eval)

    #########################################################################################
    # PHASE 2: Protein Categorization
    #########################################################################################

    ## annotate proteins
    samples_df_annotated = categorize_proteins(samples_df)

    ## aggregate protein statistics
    protein_stats_df = aggregate_protein_statistics(samples_df_annotated)

    # Save holdout samples before grouping
    cell_line_holdouts_df = samples_df_annotated[samples_df_annotated['cell_line'].isin(config['holdout_cell_lines'])].copy()

    # Remove cell line holdouts from samples_df_annotated
    samples_df_filtered = samples_df_annotated[~samples_df_annotated['cell_line'].isin(config['holdout_cell_lines'])]

    #########################################################################################
    # PHASE 3: Protein-Level Split (Two-Stage)
    #########################################################################################
    
    # Stage 1: Split train proteins to create val1 (90/10 of train proteins)
    # Ensures val1 has unique proteins not in final train
    split_dict_stage1 = create_unique_protein_split(
        samples_df_filtered,
        train_ratio=config['train_ratio_split_1'],  # 90% of stage1 train → final train, 10% → val1
        random_seed=config['random_seed']
    )

    #########################################################################################
    # PHASE 3.1: MMSEQS2 cleanup 
    #########################################################################################
    ## skip mmseqs2 cleanup for now ## 
    # cleanup_split_dict = mmseqs2_cleanup_pipeline(split_dict_stage1, config)

    # # print size of each key in cleanup_split_dict
    # for key, value in cleanup_split_dict.items():
    #     print(f"{key}: {len(value)}")

    # Stage 2: Create train/val2 split (90/10)
    # Proteins can overlap between train and val2 (different cell lines)
    split_dict_stage2 = create_stratified_splits(
        # cleanup_split_dict['train'],
        split_dict_stage1['train'],
        train_ratio=config['train_ratio_split_2'],  # 90% train, 10% val2
        random_seed=config['random_seed']
    )

    # create final split dict. consider merging val2 and train_sequence_similarity_removed, not done for now.
    split_dict = {
        'train': split_dict_stage2['train'],
        'val1': split_dict_stage1['val'],
        # 'val2': pd.concat([split_dict_stage2['val'], cleanup_split_dict['train_sequence_similarity_removed']]) ## not doing this to ensure val2 has stratified split 
        'val2': split_dict_stage2['val']
    }

    print("\n" + "="*60)
    print("FINAL SPLIT SIZES")
    print("="*60)
    print(f"Train: {len(split_dict['train'])} samples, {split_dict['train']['uniprot'].nunique()} proteins")
    print(f"Val1: {len(split_dict['val1'])} samples, {split_dict['val1']['uniprot'].nunique()} proteins")
    print(f"Val2: {len(split_dict['val2'])} samples, {split_dict['val2']['uniprot'].nunique()} proteins")
    print(f"Cell line holdouts: {len(cell_line_holdouts_df)} samples, {cell_line_holdouts_df['uniprot'].nunique()} proteins")
    # print(f"Train similarity removed: {len(cleanup_split_dict['train_sequence_similarity_removed'])} samples, {cleanup_split_dict['train_sequence_similarity_removed']['uniprot'].nunique()} proteins")

    #########################################################################################
    # PHASE 4: Save files 
    #########################################################################################

    ## summary stats 
    fig = plot_split_distributions(split_dict, plot_top_15=False)
    fig.savefig(config['output_split_distributions_png'])
    plt.close(fig)
    fig = plot_split_distributions(split_dict, plot_top_15=True)
    fig.savefig(config['output_split_distributions_top_15_png'])
    plt.close(fig)

    split_dict['train'].to_csv('train_samples.csv', index=False)
    split_dict['val1'].to_csv('val1_samples.csv', index=False)
    split_dict['val2'].to_csv('val2_samples.csv', index=False)
    cell_line_holdouts_df.to_csv('cell_line_holdouts_samples.csv', index=False)
    # cleanup_split_dict['train_sequence_similarity_removed'].to_csv('train_sequence_similarity_removed_samples.csv', index=False)

    # save images from split dicts
    print("Saving images from split dicts...")
    process_dataset_splits(
        train_samples_df = split_dict['train'], 
        val1_samples_df = split_dict['val1'], 
        val2_samples_df = split_dict['val2'], 
        cell_line_holdouts_samples_df = cell_line_holdouts_df, 
        # train_sequence_similarity_removed_samples_df = cleanup_split_dict['train_sequence_similarity_removed'], 
        samples_discovery_df = samples_df_annotated,
        output_dir = config['output_dir'], 
        config = config
    )

    # generate metadata 
    print("Generating metadata...")
    generate_metadata(
        train_samples_df = split_dict['train'], 
        val1_samples_df = split_dict['val1'], 
        val2_samples_df = split_dict['val2'], 
        cell_line_holdouts_samples_df = cell_line_holdouts_df, 
        # train_sequence_similarity_removed_samples_df = cleanup_split_dict['train_sequence_similarity_removed'], 
        output_dir = config['output_dir'], 
        config = config
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HPA Dataset Splitting with Unique Protein Constraint')
    parser.add_argument('--input_dir', type=str, help='Input directory path')
    parser.add_argument('--output_dir', type=str, help='Output directory path')
    parser.add_argument('--image_size', type=int, default=384, 
                        help='Image size for square resizing (e.g., 256 or 384). Default: 384')
    parser.add_argument('--no_mmseqs2', action='store_true', help='Skip MMSeqs2 cleanup')

    args = parser.parse_args()

    # Update config with command-line arguments
    if args.input_dir:
        config['input_dir'] = args.input_dir
    if args.output_dir:
        config['output_dir'] = os.path.join(args.output_dir, f'hpa_preprocessed_split_{args.image_size}')
    else: # if no output directory is provided, use the default output directory
        config['output_dir'] = os.path.join(config['output_dir'], f'hpa_preprocessed_split_{args.image_size}')
    if args.image_size:
        config['image_size'] = (args.image_size, args.image_size)
        print(f"Using image size: {config['image_size']}")
    if args.no_mmseqs2:
        config['run_mmseqs2'] = False

    main()