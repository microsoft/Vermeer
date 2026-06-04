# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py
#   extract_codes_c2i.py
"""
Extract VQ codes from CA (Channel-Adaptive) images.

Assumes directory structure:
data_path/train/class/img1.npy
data_path/val/class/img1.npy

saves codes and labels in the following structure:
code_path/ca{args.image_size}_codes/train/img1.npy
code_path/ca{args.image_size}_labels/train/img1.npy
code_path/ca{args.image_size}_codes/val/img1.npy
code_path/ca{args.image_size}_labels/val/img1.npy

Input: CA images with shape [H, W, 3, N_channels] stored as .npy files
Output: Token sequences with SOC/EOC markers for each channel
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
import argparse
import os
from pathlib import Path
import h5py

from utils.distributed import init_distributed_mode
from dataset.augmentation import center_crop_arr
from tokenizer.tokenizer_image.vq_model import VQ_models

from tqdm import tqdm
from PIL import Image


class CAImageDataset(Dataset):
    """
    Dataset for loading CA images (stacked multi-channel images).
    
    Each .npy file contains a tensor of shape [H, W, 3, N_channels]
    """
    def __init__(self, data_path: str, transform=None, n_channels: int = 5):
        self.data_path = Path(data_path)
        self.transform = transform
        self.n_channels = n_channels
        
        # Find all .npy files
        self.files = sorted(list(self.data_path.glob("**/*.npy")))
        if len(self.files) == 0:
            raise ValueError(f"No .npy files found in {data_path}")
        
        print(f"Found {len(self.files)} CA image files")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load CA image: [H, W, 3, N_channels]
        ca_image = np.load(self.files[idx])
        
        # Convert each channel to numpy array [H, W, 3] uint8
        channel_images = []
        # Iterate over the last dimension (channels)
        n_channels = ca_image.shape[-1]
        
        for ch_idx in range(n_channels):
            # Get channel image [H, W, 3]
            img_array = ca_image[..., ch_idx]
            
            # Ensure uint8 format (assuming values are 0-1 float or 0-255 int)
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
            channel_images.append(img_array)
        
        # Stack into single array: [N_channels, H, W, 3]
        channel_images = np.stack(channel_images, axis=0)
        
        # Get relative path
        rel_path = self.files[idx].relative_to(self.data_path)
        
        return channel_images, idx, str(rel_path)


def apply_consistent_transform(
    channel_images: np.ndarray,  # Changed type hint
    image_size: int,
    crop_range: float = 1.1,
    ten_crop: bool = False,
    flip: bool = True,
    rotate: bool = False
):
    """
    Apply the same transform consistently to all channel images.
    
    Args:
        channel_images: numpy array of shape [N_channels, H, W, 3]
        image_size: Target image size
        crop_range: Crop range for ten_crop mode
        ten_crop: Whether to use ten_crop augmentation
        flip: Whether to apply horizontal flip augmentation
        rotate: Whether to apply 4 rotation augmentations (0, 90, 180, 270 degrees)
    
    Returns:
        Tensor of shape [N_channels, num_aug, 3, H, W]
    """
    # Convert numpy arrays to PIL Images
    pil_images = [Image.fromarray(channel_images[i], mode='RGB') 
                  for i in range(channel_images.shape[0])]
    n_channels = len(pil_images)
    
    if ten_crop:
        crop_size = int(image_size * crop_range)
        # Apply center crop to all images first
        cropped_images = [center_crop_arr(img, crop_size) for img in pil_images]
        
        # TenCrop returns (top-left, top-right, bottom-left, bottom-right, center) + flipped versions
        # We need to apply the same crop index to all channels
        ten_crop_transform = transforms.TenCrop(image_size)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        all_augmented = []
        for img in cropped_images:
            crops = ten_crop_transform(img)  # Tuple of 10 PIL images
            tensors = torch.stack([normalize(to_tensor(crop)) for crop in crops])  # [10, 3, H, W]
            all_augmented.append(tensors)
        
        # Stack: [N_channels, 10, 3, H, W]
        result = torch.stack(all_augmented)
    
    else:
        crop_size = image_size
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        all_augmented = []
        for img in pil_images:
            # Center crop
            cropped = center_crop_arr(img, crop_size)
            tensor = normalize(to_tensor(cropped))  # [3, H, W]
            
            if flip:
                # Apply same flip to create augmentation
                flipped = torch.flip(tensor, dims=[-1])  # [3, H, W]
                stacked = torch.stack([tensor, flipped])  # [2, 3, H, W]
            else:
                stacked = tensor.unsqueeze(0)  # [1, 3, H, W]
            
            all_augmented.append(stacked)
        
        # Stack: [N_channels, num_aug, 3, H, W]
        result = torch.stack(all_augmented)
    
    # Apply rotation augmentation if enabled
    if rotate:
        # Apply 4 rotations (0, 90, 180, 270 degrees) consistently across channels
        # result shape: [N_channels, num_aug, 3, H, W]
        rotated = []
        for k in range(4):  # k=0: 0deg, k=1: 90deg, k=2: 180deg, k=3: 270deg
            rotated.append(torch.rot90(result, k=k, dims=[-2, -1]))
        result = torch.cat(rotated, dim=1)  # [N_channels, num_aug*4, 3, H, W]
    
    return result


def build_token_sequence(
    channel_codes: list,
    vocab_size: int = 16384,
    n_channels: int = 5
) -> np.ndarray:
    """
    Build token sequence with SOC/EOC markers.
    
    Token structure per channel: [SOC_i, codes..., EOC_i]
    Full sequence: [ch0_tokens, ch1_tokens, ..., ch4_tokens]
    
    Args:
        channel_codes: List of code arrays, one per channel, each shape (num_aug, num_patches)
        vocab_size: Base vocabulary size
        n_channels: Number of channels
    
    Returns:
        Token sequence array of shape (num_aug, total_tokens)
    """
    num_aug = channel_codes[0].shape[0]
    
    sequences = []
    for aug_idx in range(num_aug):
        seq = []
        for ch_idx in range(n_channels):
            soc_token = vocab_size + ch_idx
            eoc_token = vocab_size + n_channels + ch_idx
            codes = channel_codes[ch_idx][aug_idx]  # (num_patches,)
            
            seq.append(soc_token)
            seq.extend(codes.tolist())
            seq.append(eoc_token)
        
        sequences.append(seq)
    
    return np.array(sequences, dtype=np.int32)


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Extraction currently requires at least one GPU."
    
    # Setup DDP:
    if not args.debug:
        init_distributed_mode(args)
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    else:
        device = 'cuda'
        rank = 0
    
    # Setup output folders:
    if args.debug or rank == 0:
        os.makedirs(args.code_path, exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'ca{args.image_size}_codes'), exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'ca{args.image_size}_labels'), exist_ok=True)

    # Create and load VQ model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"Loaded VQ model from {args.vq_ckpt}")

    # Setup data:
    dataset = CAImageDataset(
        data_path=args.data_path,
        n_channels=args.n_channels
    )
    
    if not args.debug:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
            seed=args.global_seed
        )
    else:
        sampler = None
    
    loader = DataLoader(
        dataset,
        batch_size=1,  # Process one CA image at a time
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    total = 0
    for channel_images_batch, file_idxs, rel_paths in tqdm(loader):
        # channel_images_batch: [1, N_channels, H, W, 3] (batched)
        # Take first item since batch_size=1
        channel_images = channel_images_batch[0].numpy()  # [N_channels, H, W, 3]
        file_idx = file_idxs[0]
        rel_path = rel_paths[0]
        
        # Determine augmentations based on split (train vs val)
        # rel_path is relative to data_path, e.g., "train/class/img1.npy"
        path_parts = Path(rel_path).parts
        split = path_parts[0]  # 'train' or 'val'
        filename = path_parts[-1]
        is_train = split == 'train'
        
        if is_train:
            use_ten_crop = args.ten_crop
            use_flip = not args.ten_crop
            use_rotate = args.rotate
        else:
            # Validation: no augmentations (single center crop)
            use_ten_crop = False
            use_flip = False
            use_rotate = False

        # Construct new relative path: split/filename (removing class directory)
        new_rel_path = Path(split) / filename

        # Apply consistent transforms to all channels
        # Returns: [N_channels, num_aug, 3, H, W]
        augmented = apply_consistent_transform(
            channel_images,
            image_size=args.image_size,
            crop_range=args.crop_range,
            ten_crop=use_ten_crop,
            flip=use_flip,
            rotate=use_rotate
        )
        
        n_channels, num_aug, C, H, W = augmented.shape
        
        # Encode each channel
        channel_codes = []
        for ch_idx in range(n_channels):
            # Get all augmentations for this channel: [num_aug, 3, H, W]
            ch_images = augmented[ch_idx].to(device)
            
            with torch.no_grad():
                _, _, [_, _, indices] = vq_model.encode(ch_images)
            
            # indices shape: (num_aug * num_patches,) -> reshape to (num_aug, num_patches)
            num_patches = (H // 16) * (W // 16)  # VQ-16 uses 16x downsampling
            codes = indices.reshape(num_aug, num_patches)
            channel_codes.append(codes.cpu().numpy())
        
        # Build token sequence with SOC/EOC markers
        token_sequence = build_token_sequence(
            channel_codes,
            vocab_size=args.codebook_size,
            n_channels=n_channels
        )
        
        # Save codes
        save_path_code = Path(args.code_path) / f'ca{args.image_size}_codes' / new_rel_path
        # Ensure .npy extension
        save_path_code = save_path_code.with_suffix('.npy')
        save_path_code.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(
            save_path_code,
            token_sequence  # Shape: (num_aug, total_tokens)
        )
        
        if args.label_file is not None:
            if args.label_type is not None and args.label_type in h5py.File(args.label_file, 'r'):
                if args.label_type == 'localization_onehot':
                    label_one_hot = h5py.File(args.label_file, 'r')[args.label_type][filename]
                    label = np.array(label_one_hot, dtype=np.float32).squeeze() # shape (1, 31) 
                elif args.label_type == 'esm_embed_full':
                    esm_embed = h5py.File(args.label_file, 'r')[args.label_type][filename]
                    label = np.array(esm_embed, dtype=np.float32).squeeze() # shape (1, L+2, 1152) 
                    if np.isnan(label).sum() > 0:
                        print(f"Warning: NaN detected in esm_embed_full label for {filename}")
                elif args.label_type == 'esm_embed_mean_pool':
                    esm_embed = h5py.File(args.label_file, 'r')[args.label_type][filename]
                    label = np.array(esm_embed, dtype=np.float32).squeeze() # shape (1152,) 
                    if np.isnan(label).sum() > 0:
                        print(f"Warning: NaN detected in esm_embed_mean_pool label for {filename}")
                else:
                    raise ValueError(f"Label type {args.label_type} not supported")
            else:
                raise ValueError(f"Label type {args.label_type} not found in {args.label_file}")
        else: # Save -1 as label for unconditional case
            label = np.array([-1])
        save_path_label = Path(args.code_path) / f'ca{args.image_size}_labels' / new_rel_path
        # Ensure .npy extension
        save_path_label = save_path_label.with_suffix('.npy')
        save_path_label.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(
            save_path_label,
            label
        )
        
        if not args.debug:
            total += dist.get_world_size()
        else:
            total += 1
    
    if not args.debug:
        dist.destroy_process_group()
    
    print(f"Extraction complete. Processed {total} CA images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to directory containing CA .npy files")
    parser.add_argument("--code-path", type=str, required=True,
                        help="Path to save extracted codes")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True,
                        help="Checkpoint path for VQ model")
    parser.add_argument("--codebook-size", type=int, default=16384,
                        help="Codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8,
                        help="Codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--n-channels", type=int, default=5,
                        help="Number of channels in CA images")
    parser.add_argument("--ten-crop", action='store_true',
                        help="Whether to use ten-crop augmentation")
    parser.add_argument("--crop-range", type=float, default=1.1,
                        help="Expanding range of center crop for ten-crop")
    parser.add_argument("--rotate", action='store_true',
                        help="Whether to apply 4 rotation augmentations (0, 90, 180, 270 degrees)")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--debug", action='store_true',
                        help="Run in debug mode (single GPU, no DDP)")
    parser.add_argument("--label-type", type=str,
                    help="Type of label, choices are ['localization_onehot', 'esm_embed_full', 'esm_embed_mean_pool]")
    parser.add_argument("--label-file", type=str,
                    help="Path to h5 file containing labels, must be a h5 file with label-type as a key")
    args = parser.parse_args()
    main(args)

