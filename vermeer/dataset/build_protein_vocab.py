"""
Build the protein vocab bundle for the `ca_learnable_protein_embed` conditioning mode.

Scans the TRAIN codes dir for unique UniProt ids (the first underscore-separated
token of each code filename), assigns contiguous indices in sorted order, and gathers
each train protein's ESM mean-pool vector from the matching label file. Writes:

  {out}/uniprot_to_index.json : {uniprot_id: int_index}
  {out}/train_ids.json        : [uid_0 ... uid_{N-1}]  (row order of the matrix)
  {out}/train_esm.npy         : (N, 1152) float32 ESM mean-pool matrix

Build ONCE from the train split and reuse the same bundle for every split / for the
life of a checkpoint (the embedding table is tied to this index ordering).

Example:
  python dataset/build_protein_vocab.py \
      --code-path /path/to/hpa_..._code_flip_ten_crop_rotate \
      --image-size 256 \
      --split train \
      --out /path/to/hpa_..._code_flip_ten_crop_rotate/ca256_protein_vocab
"""
import os
import json
import argparse
import numpy as np

from dataset.protein_vocab import (
    extract_uniprot_id,
    UNIPROT_TO_INDEX_FILE,
    TRAIN_IDS_FILE,
    TRAIN_ESM_FILE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-path', type=str, required=True,
                        help='Root containing ca{image_size}_codes and ca{image_size}_labels')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--split', type=str, default='train',
                        help='Split to build the vocab from (should be the TRAIN split)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output bundle dir (default: {code_path}/ca{image_size}_protein_vocab)')
    args = parser.parse_args()

    codes_dir = os.path.join(args.code_path, f'ca{args.image_size}_codes', args.split)
    labels_dir = os.path.join(args.code_path, f'ca{args.image_size}_labels', args.split)
    assert os.path.isdir(codes_dir), f'Missing codes dir: {codes_dir}'
    assert os.path.isdir(labels_dir), f'Missing labels dir: {labels_dir}'

    out_dir = args.out or os.path.join(args.code_path, f'ca{args.image_size}_protein_vocab')
    os.makedirs(out_dir, exist_ok=True)

    # First occurrence per UniProt id (the ESM vector is identical across an id's files).
    uid_to_file = {}
    for fname in sorted(os.listdir(codes_dir)):
        if not fname.endswith('.npy'):
            continue
        uid = extract_uniprot_id(fname)
        if uid not in uid_to_file:
            uid_to_file[uid] = fname

    ids = sorted(uid_to_file.keys())
    uniprot_to_index = {uid: i for i, uid in enumerate(ids)}
    print(f'Found {len(ids)} unique proteins in {codes_dir}')

    esm_rows = []
    n_nan = 0
    for uid in ids:
        label = np.load(os.path.join(labels_dir, uid_to_file[uid])).astype(np.float32).squeeze()
        if label.ndim != 1:
            raise ValueError(
                f'Expected a 1-D ESM mean-pool label for {uid_to_file[uid]}, got shape {label.shape}. '
                f'Build the vocab from codes/labels extracted with --label-type esm_embed_mean_pool.')
        if np.isnan(label).any():
            n_nan += 1
        esm_rows.append(label)

    train_esm = np.stack(esm_rows).astype(np.float32)  # (N, 1152)
    if n_nan:
        print(f'Warning: {n_nan} proteins have NaN ESM vectors (e.g. empty sequence); '
              f'they are kept with their own learned row but excluded from NN matches.')

    with open(os.path.join(out_dir, UNIPROT_TO_INDEX_FILE), 'w') as f:
        json.dump(uniprot_to_index, f)
    with open(os.path.join(out_dir, TRAIN_IDS_FILE), 'w') as f:
        json.dump(ids, f)
    np.save(os.path.join(out_dir, TRAIN_ESM_FILE), train_esm)

    print(f'Wrote vocab bundle to {out_dir}')
    print(f'  num_proteins = {len(ids)}, train_esm.npy shape = {train_esm.shape}')


if __name__ == '__main__':
    main()
