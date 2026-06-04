"""
CA (Channel-Augmented) Image Code Dataset for training gpt_ca models.

Loads pre-extracted VQ codes with SOC/EOC markers and appends EOS for training.
"""
import torch
import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset


class CACodeDataset(Dataset):
    """
    Dataset for loading CA image codes with SOC/EOC markers.
    
    Each code file contains shape (num_aug, seq_len) where:
    - num_aug: number of augmentations (2 for flip, 10 for ten_crop)
    - seq_len: n_channels * (1 + patches_per_channel + 1) = n_channels * 258
    
    The dataset appends EOS token to create targets for training.
    """
    def __init__(
        self, 
        feature_dir: str, 
        label_dir: str,
        vocab_size: int = 16384,
        n_channels: int = 5,
        model_type: str = 'ca'
    ):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.vocab_size = vocab_size
        self.n_channels = n_channels
        self.flip = 'flip' in self.feature_dir
        self.model_type = model_type
        # EOS token ID: vocab_size + 2 * n_channels
        self.eos_token_id = vocab_size + 2 * n_channels
        
        # Check for additional augmentation directory
        aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        self.feature_files = sorted(os.listdir(feature_dir))
        self.label_files = sorted(os.listdir(label_dir))

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        # Optionally use augmented features
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir
                   
        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        # Load features: shape (num_aug, seq_len)
        # seq_len = n_channels * (1 + patches_per_channel + 1)
        features = np.load(os.path.join(feature_dir, feature_file))
        
        # Randomly select an augmentation
        if len(features.shape) > 1 and features.shape[0] > 1:
            aug_idx = torch.randint(low=0, high=features.shape[0], size=(1,)).item()
            features = features[aug_idx]  # (seq_len,)
        elif len(features.shape) > 1:
            features = features[0]  # (seq_len,)
        
        # Append EOS token to create the full sequence
        # Input to model will be features (without EOS)
        # Target will be features shifted by 1 (with EOS at end)
        features_with_eos = np.concatenate([features, [self.eos_token_id]])
        
        # Load labels
        labels = np.load(os.path.join(label_dir, label_file))
        
        # ensuring model_type is valid, handled in train_ca.py args 
        if self.model_type == 'ca' or self.model_type == 'ca_binary_prefix':
            labels_tensor = torch.from_numpy(labels).long()
        elif self.model_type == 'ca_esm_embed_mean_pool' or self.model_type == 'ca_esm_embed_full':
            labels_tensor = torch.from_numpy(labels).float()
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

        return (
            torch.from_numpy(features_with_eos).long(),
            labels_tensor,
            self.n_channels
        )


def build_ca_code(args, split='train'):
    """Build CA code dataset from extracted codes."""
    feature_dir = os.path.join(args.code_path, f"ca{args.image_size}_codes", split)
    label_dir = os.path.join(args.code_path, f"ca{args.image_size}_labels", split)
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"Please first run extract_codes_ca.py to create {feature_dir} and {label_dir}"
    
    return CACodeDataset(
        feature_dir=feature_dir,
        label_dir=label_dir,
        vocab_size=args.vocab_size,
        n_channels=args.n_channels,
        model_type=args.gpt_type
    )

