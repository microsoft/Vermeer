"""
Prepare a Protein Prefix H5 Files. Supported prefixes are one-hot (localization category) and ESM-C embedding.

This script adds one-hot (localization category) by referencing an input metadata csv file.
This script adds ESM-C 600M embeddings by extracting UniProt IDs from filenames, fetching sequences, and generating embeddings.
Mean-pooled, cls, and full embeddings are saved to the output h5 file under different keys. 
"""

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

import torch
import os
import glob
import argparse
import h5py
import requests
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import time

# Global client (initialized lazily)
_client = None

# as discussed and demonstrated in the following 2 links, disable autocast and explicitly use float32 to avoid precision issues
# https://github.com/ziul-bio/SWAT/blob/main/scripts/extract_ESMC.py
# https://www.nature.com/articles/s41598-025-05674-x

# todo: add batching for efficiency (https://huggingface.co/Synthyra/ESMplusplus_large) 
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
    30: "Unknown",
}

def get_client(device="cuda"):
    """Get or initialize the ESM-C client."""
    global _client
    if _client is None:
        print(f"Loading ESM-C 600M model on {device}...")
        _client = ESMC.from_pretrained("esmc_600m").to(device)
        _client = _client.to(torch.float32) # explicitly cast to float32 to avoid precision issues
    return _client

def embed_sequence(seq, device="cuda"):
    """Generate ESM-C embedding for a single sequence (FASTA format)."""
    client = get_client(device)
    protein = ESMProtein(seq)

    with torch.no_grad():
        with torch.autocast(device_type='cuda', enabled=False):
            protein_tensor = client.encode(protein)
            logits_output = client.logits(
                protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )
            protein_embedding = logits_output.embeddings  # (L+2, dim) bc of cls token and EOS token
            cls_token_emb = protein_embedding[0][0].squeeze()
            protein_embedding = protein_embedding.squeeze()[1:-1] # remove cls and eos tokens
            mean_pooled_embedding = extract_mean_representation(protein_embedding)
            
    return protein_embedding.detach().cpu().numpy(), mean_pooled_embedding.detach().cpu().numpy(), cls_token_emb.detach().cpu().numpy()

def extract_mean_representation(protein_embedding):
    """Extract mean representation from protein embedding."""
    return protein_embedding.squeeze()[1:-1].mean(axis=0)

def get_uniprot_protein_sequence(uniprot_id):
    """Fetch protein sequence from UniProt API."""
    seq_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    seq_response = requests.get(seq_url, timeout=30)

    if not seq_response.ok:
        max_retries = 5
        for attempt in range(max_retries):
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"[WARNING] Failed to fetch sequence for {uniprot_id}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
            seq_response = requests.get(seq_url, timeout=30)
            lines = seq_response.text.strip().split('\n')
            if seq_response.ok:
                lines = seq_response.text.strip().split('\n')
                seq = ''.join(lines[1:])
                return seq
        print(f"[ERROR] Failed to fetch sequence for {uniprot_id}")
        return None

    lines = seq_response.text.strip().split('\n')
    seq = ''.join(lines[1:])
    return seq

def get_sequence_from_uniprot(uniprot_id):
    """Fetch protein amino-acid sequence from UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Could not fetch sequence for {uniprot_id} (HTTP {response.status_code})")
    lines = response.text.strip().split('\n')
    return ''.join(lines[1:])  # skip FASTA header

def extract_uniprot_id(filename):
    """
    Extract UniProt ID from filename.
    
    Example: Q5VZP5_HPA012912_U2OS_fov_2.npy -> Q5VZP5
    """
    basename = os.path.basename(filename)
    return basename.split("_")[0]


def collect_filenames_from_dirs(sub_dirs):
    """
    Collect all filenames from train and val directories.
    
    Args:
        sub_dirs: List of paths to data directories (e.g., train, val)
        
    Returns:
        Set of filenames (basenames only)
    """
    filenames = set()
    
    for data_dir in sub_dirs:
        if os.path.exists(data_dir):
            print(f"Searching for .npy files in {data_dir}...")
            # Search for .npy files in subdirectories like 0000/
            npy_files = glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)
            for f in npy_files:
                filenames.add(os.path.basename(f))
    
    return filenames

def collect_filenames(sub_dirs):

    filenames = collect_filenames_from_dirs(sub_dirs)
    
    if not filenames:
        raise ValueError("No filenames found directories")
    
    print(f"Found {len(filenames)} files to process")
    return filenames

def build_uniprot_to_filenames(filenames):
    # Build mapping from UniProt ID to filenames
    uniprot_to_filenames = defaultdict(list)
    for filename in filenames:
        uniprot_id = extract_uniprot_id(filename)
        uniprot_to_filenames[uniprot_id].append(filename)
    
    unique_uniprots = list(uniprot_to_filenames.keys())
    print(f"Found {len(unique_uniprots)} unique UniProt IDs")

    return uniprot_to_filenames, unique_uniprots

# TODO: add batching for efficiency to uniprot download and esm embedding
def fetch_and_embed_sequences(unique_uniprots, output_embeddings_h5, device="cuda"):
    """Fetch sequences and generate embeddings, streaming to disk."""
    failed_uniprots = []
    
    print("\nFetching sequences and generating embeddings...")
    with h5py.File(output_embeddings_h5, 'w') as f_emb:
        # Create groups for different embedding types
        full_grp = f_emb.create_group('esm_embed_full')
        mean_pool_grp = f_emb.create_group('esm_embed_mean_pool')
        
        for uniprot_id in tqdm(unique_uniprots, desc="Processing UniProt IDs"):
            try:
                fasta_sequence = get_uniprot_protein_sequence(uniprot_id)
                if fasta_sequence is None:
                    failed_uniprots.append(uniprot_id)
                    continue
                
                protein_embedding, mean_pooled_embedding, cls_token_emb = embed_sequence(fasta_sequence, device=device)
                # Write immediately to disk, don't keep in memory
                full_grp.create_dataset(uniprot_id, data=protein_embedding)
                mean_pool_grp.create_dataset(uniprot_id, data=mean_pooled_embedding)
                
            except Exception as e:
                print(f"\n[ERROR] Failed to process {uniprot_id}: {e}")
                failed_uniprots.append(uniprot_id)
    
    print(f"\nSuccessfully embedded {len(unique_uniprots) - len(failed_uniprots)}/{len(unique_uniprots)} proteins")
    if failed_uniprots:
        print(f"Failed UniProt IDs: {failed_uniprots[:10]}{'...' if len(failed_uniprots) > 10 else ''}")
    
    return output_embeddings_h5  # Return path to embeddings file

def write_esm_embeddings_to_h5(h5_path, filenames, embeddings_h5_path):
    """Write final h5 file, reading embeddings from intermediate file."""
    with h5py.File(embeddings_h5_path, 'r') as f_emb, \
         h5py.File(h5_path, 'w') as f_out:
        
        # Add esm_embed group (full embeddings)
        print("  Creating group: esm_embed")
        esm_grp = f_out.create_group('esm_embed_full')
        
        for filename in tqdm(filenames, desc="  Writing esm_embed"):
            uniprot_id = extract_uniprot_id(filename)
            if uniprot_id in f_emb['esm_embed_full']:
                embedding = f_emb['esm_embed_full'][uniprot_id][:]
                esm_grp.create_dataset(filename, data=embedding)
        
        # Add esm_embed_mean_pool group
        print("  Creating group: esm_embed_mean_pool")
        esm_mean_grp = f_out.create_group('esm_embed_mean_pool')
        
        for filename in tqdm(filenames, desc="  Writing esm_embed_mean_pool"):
            uniprot_id = extract_uniprot_id(filename)
            if uniprot_id in f_emb['esm_embed_mean_pool']:
                mean_embedding = f_emb['esm_embed_mean_pool'][uniprot_id][:]
                esm_mean_grp.create_dataset(filename, data=mean_embedding)
    
    print(f"\nDone! Output saved to: {h5_path}")

def get_localization_from_csv(filename, metadata_df):
    """
    Get localization list for a given filename from metadata CSV.
    
    Args:
        filename: Filename (e.g., 'Q5VZP5_HPA012912_U2OS_fov_2.npy')
        metadata_df: DataFrame with metadata including 'base_filename' and 'localizations'
        
    Returns:
        List of localization strings, or None if not found
    """
    # Remove .npy extension if present
    base_filename = filename.replace('.npy', '')
    
    # Look up in metadata dataframe
    matches = metadata_df[metadata_df['base_filename'] == base_filename]
    
    if len(matches) > 0:
        localizations = matches.iloc[0]['localizations']
        # If it's a string (from CSV), parse it
        if isinstance(localizations, str):
            try:
                return ast.literal_eval(localizations)
            except:
                return None
        # If already a list
        elif isinstance(localizations, list):
            return localizations
    
    return None

def collect_filenames_from_h5(h5_path):
    """
    Collect all filenames from an existing h5 file.
    
    Args:
        h5_path: Path to the h5 file
        
    Returns:
        Set of filenames
    """
    filenames = set()
    with h5py.File(h5_path, 'r') as f:
        # Try to get filenames from existing keys (binary, one_hot, esm_embed)
        for key in f.keys():
            if key in ['localization_onehot', 'esm_embed_full', 'esm_embed_mean_pool']:
                filenames.update(f[key].keys())
    return filenames

def write_localization_category_to_h5(input_dir, metadata_dir, output_path):
    """
    Write localization category to h5 file as multi-label binary vectors.
    
    Args:
        input_dir: Directory containing train, val1, val2, etc. subdirectories
        metadata_dir: Directory containing metadata CSV files
        output_path: Path for the new h5 file with localization data
    """

    # Get full paths to subdirectories (only actual directories, not files)
    sub_dirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir) 
                if os.path.isdir(os.path.join(input_dir, d))]
    
    print("\n" + "="*60)
    print("WRITING LOCALIZATION CATEGORIES TO H5")
    print("="*60)
    
    # Load metadata CSV files
    csv_files = glob.glob(os.path.join(metadata_dir, "*_samples.csv"))
    print(f"Loading metadata from {metadata_dir}...")
    print(f"Found {len(csv_files)} CSV files")
    
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        print(f"  Loaded {len(df)} records from {os.path.basename(csv_file)}")
        dfs.append(df)
    
    metadata_df = pd.concat(dfs, ignore_index=True)
    print(f"Total records loaded: {len(metadata_df)}")
    
    # Parse localizations column if it's stored as string
    if 'localizations' in metadata_df.columns and isinstance(metadata_df['localizations'].iloc[0], str):
        print("Parsing localizations column...")
        metadata_df['localizations'] = metadata_df['localizations'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    
    # Get valid classes from CLASS2NAME and create lowercase mapping
    print("\nPreparing localization class mapping...")
    valid_classes = set(CLASS2NAME.values())
    valid_classes_lower = {cls.lower(): cls for cls in valid_classes}
    
    # Infer all unique localization classes from metadata
    print("\nInferring localization classes from metadata...")
    all_locs_in_metadata = set()
    for locs in metadata_df['localizations']:
        if isinstance(locs, list):
            all_locs_in_metadata.update(locs)
    
    print(f"Found {len(all_locs_in_metadata)} unique localization classes in metadata")
    
    # Check for missing classes
    missing_classes = set()
    for loc in all_locs_in_metadata:
        loc_lower = loc.lower()
        if loc_lower not in valid_classes_lower:
            missing_classes.add(loc)
    
    if missing_classes:
        print(f"\nWARNING: Found {len(missing_classes)} localization classes in metadata NOT in CLASS2NAME:")
        for loc in sorted(missing_classes):
            print(f"  - {loc}")
        print("  These will be classified as 'unknown'")
    
    # Use all classes from CLASS2NAME (sorted for consistent ordering)
    all_classes = sorted(valid_classes)
    print(f"\nUsing {len(all_classes)} classes from CLASS2NAME")
    
    # Collect filenames from directories
    print(f"\nScanning directories: {sub_dirs}")
    filenames = collect_filenames_from_dirs(sub_dirs)
    print(f"Found {len(filenames)} .npy files in directories")
    
    # Build filename to localization mapping
    print("\nBuilding filename to localization mapping...")
    
    # Collect all localizations for each file
    filename_list = []
    localization_lists = []
    missing_files = []
    
    for filename in tqdm(filenames, desc="Collecting localizations"):
        localizations = get_localization_from_csv(filename, metadata_df)
        
        if localizations is None:
            missing_files.append(filename)
            # Empty list for missing files (will result in zero vector)
            localizations = []
        
        # Normalize localization names (case-insensitive matching with CLASS2NAME)
        normalized_locs = []
        for loc in localizations:
            loc_lower = loc.lower()
            # Find matching class in CLASS2NAME (case-insensitive)
            if loc_lower in valid_classes_lower:
                normalized_locs.append(valid_classes_lower[loc_lower])
            else:
                # If not found, label as Unknown
                normalized_locs.append("unknown")
        
        filename_list.append(filename)
        localization_lists.append(normalized_locs)
    
    # Use MultiLabelBinarizer to create binary vectors
    print("\nEncoding localizations with MultiLabelBinarizer...")
    mlb = MultiLabelBinarizer(classes=all_classes)
    # print(f"MultiLabelBinarizer classes: {all_classes}")
    localization_matrix = mlb.fit_transform(localization_lists)
    
    # Convert to float32 and create dictionary mapping filename to vector
    localization_vectors = {}
    for filename, label_vector in zip(filename_list, localization_matrix):
        localization_vectors[filename] = label_vector.astype(np.float32)
    
    num_classes = len(all_classes)
    
    # Write to new h5 file
    print(f"\nWriting localization vectors to {output_path}...")
    with h5py.File(output_path, 'w') as f_out:
        loc_grp = f_out.create_group('localization_onehot')
        
        for filename in tqdm(localization_vectors.keys(), desc="Writing to h5"):
            loc_grp.create_dataset(filename, data=localization_vectors[filename])
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total samples processed: {len(localization_vectors)}")
    print(f"Missing files in CSV: {len(missing_files)}")
    if missing_files[:5]:
        print(f"  Sample missing files: {missing_files[:5]}")
    
    # Calculate average number of localizations per sample
    non_zero_samples = [v for v in localization_vectors.values() if v.sum() > 0]
    if non_zero_samples:
        avg_locs = np.mean([v.sum() for v in non_zero_samples])
        print(f"\nAverage localizations per sample (non-zero): {avg_locs:.2f}")
    
    # Class distribution
    all_vectors = np.array(list(localization_vectors.values()))
    class_counts = all_vectors.sum(axis=0)
    
    print(f"\nClass distribution (top 10):")
    # MultiLabelBinarizer uses alphabetical ordering of classes
    # mlb.classes_ contains the ordered class names
    top_indices = np.argsort(class_counts)[-10:][::-1]
    for idx in top_indices:
        class_name = mlb.classes_[idx]
        print(f"  {class_name}: {int(class_counts[idx])} samples")
    
    print(f"\nDone! Output saved to: {output_path}")
    print(f"Output shape: ({len(localization_vectors)}, {num_classes})")
    print("="*60)


def add_esm_embeddings_to_h5(input_dir, h5_filename, device="cuda", skip_cache=False):
    """
    Add ESM-C embeddings to an existing protein prefix h5 file.

    Args:
        input_dir: Directory containing protein_prefix.h5, train/, and val/ directories
        h5_filename: Name of the input h5 file
        device: Device to run ESM-C model on (cuda or cpu)
        skip_cache: If True, force regeneration of embeddings even if cached file exists
    """
    # Paths
    h5_path = os.path.join(input_dir, h5_filename)
    # Get full paths to subdirectories (only actual directories, not files)
    sub_dirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir) 
                if os.path.isdir(os.path.join(input_dir, d))]
    
    # Validate inputs
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    print(f"Input H5 file: {h5_path}")
    print(f"Sub directories: {sub_dirs}")
    
    filenames = collect_filenames_from_dirs(sub_dirs)
    uniprot_to_filenames, unique_uniprots = build_uniprot_to_filenames(filenames)

    # Intermediate file for streaming embeddings
    embeddings_h5_path = os.path.join(input_dir, "uniprot_embeddings_temp.h5")
    
    if skip_cache and os.path.exists(embeddings_h5_path):
        print(f"Skipping cache: removing existing embeddings file {embeddings_h5_path}")
        os.remove(embeddings_h5_path)
    
    if not os.path.exists(embeddings_h5_path):
        # Stream embeddings to disk instead of keeping in memory
        fetch_and_embed_sequences(unique_uniprots, embeddings_h5_path, device=device)
    else:
        print(f"Intermediate embeddings file already exists: {embeddings_h5_path}")
    
    # TODO: add assert that all the filenames are aligned across groups
    
    # write to tmp intermediate h5 file
    write_esm_embeddings_to_h5(h5_path, filenames, embeddings_h5_path)
    
    # Print summary
    with h5py.File(h5_path, 'r') as f:
        print("\nOutput file structure:")
        for key in f.keys():
            n_entries = len(f[key].keys())
            sample_key = list(f[key].keys())[0] if n_entries > 0 else None
            if sample_key:
                sample_shape = f[key][sample_key].shape
                print(f"  {key}: {n_entries} entries, sample shape: {sample_shape}")
            else:
                print(f"  {key}: {n_entries} entries")


def main():
    parser = argparse.ArgumentParser(
        description="Add ESM-C embeddings to protein prefix H5 file"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing protein_prefix.h5, train/, and val/"
    )
    parser.add_argument(
        "--h5_filename",
        type=str,
        default="protein_prefix.h5",
        help="Name of input h5 file (default: protein_prefix.h5)"
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        required=True,
        help="Directory containing metadata CSV files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run ESM-C model on (default: cuda)"
    )
    parser.add_argument(
        "--skip_cache",
        action="store_true",
        help="Skip cache and force regeneration of embeddings even if cached file exists"
    )
    
    args = parser.parse_args()

    write_localization_category_to_h5(
        input_dir=args.input_dir,
        metadata_dir=args.metadata_dir,
        output_path=os.path.join(args.input_dir, args.h5_filename)
    )
    
    add_esm_embeddings_to_h5(
        input_dir=args.input_dir,
        h5_filename=args.h5_filename,
        device=args.device,
        skip_cache=args.skip_cache
    )


if __name__ == "__main__":
    main()