"""
Shared protein vocabulary + nearest-neighbor resolver for the
`ca_learnable_protein_embed` conditioning mode.

A vocab bundle (built once from the TRAIN split by `build_protein_vocab.py`) is a
directory containing:
  - uniprot_to_index.json : {uniprot_id: int_index} for train proteins (0..N-1)
  - train_ids.json        : ordered [uid_0 ... uid_{N-1}] (row order of the matrix)
  - train_esm.npy         : (N, 1152) float32 train ESM mean-pool matrix

Seen (train) proteins map to their own learned row. Unseen proteins (e.g. val1 OOD)
are resolved at runtime to the nearest train protein by cosine similarity in ESM
mean-pool space, and that train row's index is used. Index `num_proteins` is reserved
as the learnable null / unconditional protein (handled inside ProteinLookupEmbedder).
"""
import os
import json
import numpy as np
import torch


def extract_uniprot_id(filename):
    """UniProt id is the first underscore-separated token of the filename.

    e.g. Q13950_HPA022040_A-431_fov_1.npy -> Q13950
    """
    return os.path.basename(filename).split('_')[0]


UNIPROT_TO_INDEX_FILE = 'uniprot_to_index.json'
TRAIN_IDS_FILE = 'train_ids.json'
TRAIN_ESM_FILE = 'train_esm.npy'


class ProteinVocab:
    """Loads a vocab bundle and resolves UniProt ids -> integer protein index."""

    def __init__(self, vocab_dir):
        self.vocab_dir = vocab_dir
        with open(os.path.join(vocab_dir, UNIPROT_TO_INDEX_FILE)) as f:
            self.uniprot_to_index = json.load(f)
        self.num_proteins = len(self.uniprot_to_index)

        with open(os.path.join(vocab_dir, TRAIN_IDS_FILE)) as f:
            self.train_ids = json.load(f)

        train_esm = np.load(os.path.join(vocab_dir, TRAIN_ESM_FILE)).astype(np.float32)  # (N, 1152)
        train_esm = np.nan_to_num(train_esm, nan=0.0)  # NaN proteins (e.g. empty seq) -> zero row
        t = torch.from_numpy(train_esm)
        # Pre-normalize for cosine similarity; zero rows stay zero (never selected).
        self.train_esm_norm = t / t.norm(dim=1, keepdim=True).clamp_min(1e-8)  # (N, 1152)
        self._nn_cache = {}

    def num_proteins_total(self):
        return self.num_proteins

    def index_for(self, uniprot_id, esm_vec=None):
        """Return the integer protein index for `uniprot_id`.

        Seen proteins return their own index. Unseen proteins require `esm_vec`
        (their ESM mean-pool embedding) and are resolved to the nearest train
        protein by cosine similarity (cached per uid).
        """
        idx = self.uniprot_to_index.get(uniprot_id)
        if idx is not None:
            return idx
        if uniprot_id in self._nn_cache:
            return self._nn_cache[uniprot_id]
        if esm_vec is None:
            raise ValueError(
                f"Unseen protein '{uniprot_id}' requires an ESM mean-pool vector for "
                f"nearest-neighbor resolution, but none was provided.")
        q = esm_vec if torch.is_tensor(esm_vec) else torch.from_numpy(np.asarray(esm_vec))
        q = q.float().reshape(-1)
        q = torch.nan_to_num(q, nan=0.0)
        q = q / q.norm().clamp_min(1e-8)
        nn_idx = int((self.train_esm_norm @ q).argmax())
        self._nn_cache[uniprot_id] = nn_idx
        return nn_idx
