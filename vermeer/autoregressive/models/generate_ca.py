# Modified from:
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo.config
import torch._inductor.config
import copy
import os
import numpy as np
from pathlib import Path
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def logits_to_probs(logits, temperature: float = 1.0, top_p: float=1.0, top_k: int = None, **kwargs):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def prefill(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    if cfg_scale > 1.0:
        logits, _ = model(None, cond_idx, input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = model(None, cond_idx, input_pos)

    return sample(logits, **sampling_kwargs)[0]

def prefill_with_prefix(
    model, 
    cond_idx: torch.Tensor,
    prefix_tokens: torch.Tensor,
    cfg_scale: float = 1.0,
    padding_mask: torch.Tensor = None,
    lens: torch.Tensor = None,
    **sampling_kwargs
) -> Tuple[torch.Tensor, int]:
    """
    Prefill KV cache with conditioning token and image prefix tokens.
    """
    device = cond_idx.device
    prefix_len = prefix_tokens.shape[1]

    input_pos_cond_prefix = torch.arange(0,1+prefix_len, device=device, dtype=torch.int)

    if cfg_scale > 1.0:
        logits, _ = model(prefix_tokens, cond_idx=cond_idx, input_pos=input_pos_cond_prefix, non_mask=padding_mask, lens=lens)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = model(prefix_tokens, cond_idx=cond_idx, input_pos=input_pos_cond_prefix, non_mask=padding_mask, lens=lens)
    
    next_token = sample(logits, **sampling_kwargs)[0]
    next_pos = 1 + prefix_len 
    
    return next_token, next_pos


def decode_one_token(model, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, padding_mask: torch.Tensor = None, lens: torch.Tensor = None, **sampling_kwargs):
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits, _ = model(x_combined, cond_idx=None, input_pos=input_pos, non_mask=padding_mask, lens=lens)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits, _ = model(x, cond_idx=None, input_pos=input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int,
    padding_mask: torch.Tensor = None,
    lens: torch.Tensor = None,
    **sampling_kwargs):
    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, cfg_scale, cfg_flag, padding_mask=padding_mask, lens=lens, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(-1, 1)
    
    return new_tokens, new_probs

@torch.no_grad()
def generate(model, cond, max_new_tokens, emb_masks=None, cfg_scale=1.0, cfg_interval=-1, **sampling_kwargs):
    # if model.model_type == 'c2i':
    #     if cfg_scale > 1.0:
    #         cond_null = torch.ones_like(cond) * model.num_classes
    #         cond_combined = torch.cat([cond, cond_null])
    #     else:
    #         cond_combined = cond
    #     T = 1
    # elif model.model_type == 't2i':
    #     if cfg_scale > 1.0:
    #         cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
    #         cond_combined = torch.cat([cond, cond_null])
    #     else:
    #         cond_combined = cond
    #     T = cond.shape[1]      
    # else:
    #     raise Exception("please check model type")

    if model.model_type == 'ca':
        if cfg_scale > 1.0:
            raise Exception("CFG is not supported for CA model because it is an unconditional model")
        else:
            cond_combined = cond
    elif model.model_type == 'ca_binary_prefix':
        if cfg_scale > 1.0:
            raise NotImplementedError("CFG is not implemented for CA_binary_prefix model")
        else:
            cond_combined = cond
    elif model.model_type == 'ca_esm_embed_mean_pool':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
    elif model.model_type == 'ca_esm_embed_full':
        # raise NotImplementedError("generation not implemented for CA_esm_embed_full model")
        ## TODO: finish testing this
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
    else:
        raise Exception("please check model type")

    T = cond_combined.shape[1]

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings_extended.weight.dtype)
    
    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, T:T+1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(model, next_token, input_pos, max_new_tokens-1, cfg_scale, cfg_interval, **sampling_kwargs)
    seq[:, T+1:] = torch.cat(generated_tokens, dim=1)

    return seq[:, T:]

@torch.no_grad()
def generate_with_prefix(
    model,
    cond_idx: torch.Tensor,
    prefix_tokens: torch.Tensor,
    n_total_channels: int,
    tokens_per_channel: int = 258,
    cfg_scale: float = 1.0,
    cfg_interval: int = -1,
    padding_mask: torch.Tensor = None,
    lens: torch.Tensor = None,
    return_logps: bool = False,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Generate remaining channels given prefix channels.
    
    Sequence structure:
    [cond_token, SOC_0, patches_ch0..., EOC_0, SOC_1, patches_ch1..., EOC_1, ..., EOS]
    """

    batch_size = cond_idx.shape[0]
    device = cond_idx.device
    prefix_len = prefix_tokens.shape[1]

    if model.model_type == 'ca':
        if cfg_scale > 1.0:
            raise Exception("CFG is not supported for CA model because it is an unconditional model")
        else:
            cond_combined = cond_idx
            prefix_combined = prefix_tokens
    elif model.model_type == 'ca_binary_prefix':
        if cfg_scale > 1.0:
            raise NotImplementedError("CFG is not implemented for CA_binary_prefix model")
        else:
            cond_combined = cond_idx
            prefix_combined = prefix_tokens
    elif model.model_type == 'ca_esm_embed_mean_pool':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond_idx) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond_idx, cond_null])
            prefix_combined = torch.cat([prefix_tokens, prefix_tokens]) # duplicate prefix tokens for cfg 
        else:
            cond_combined = cond_idx
            prefix_combined = prefix_tokens
    elif model.model_type == 'ca_esm_embed_full':
        if cfg_scale > 1.0:
            # raise NotImplementedError("CFG is not implemented for CA_esm_embed_full model")
            cond_null = torch.zeros_like(cond_idx) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond_idx, cond_null])
            prefix_combined = torch.cat([prefix_tokens, prefix_tokens]) # duplicate prefix tokens for cfg 
        else:
            cond_combined = cond_idx
            prefix_combined = prefix_tokens
    else:
        raise Exception("please check model type")
    
    # Calculate sequence lengths
    # Total tokens = n_total_channels * tokens_per_channel + 1 (EOS)
    total_tokens = n_total_channels * tokens_per_channel + 1
    # Number of tokens to generate = total_tokens - prefix_len
    num_new_tokens = total_tokens - prefix_len
    
    # Total sequence length including condition token
    max_seq_length = 1 + total_tokens  # 1 for cond_token
    
    # Setup KV caches
    with torch.device(device):
        max_batch_size_cfg = batch_size * 2 if cfg_scale > 1.0 else batch_size
        model.setup_caches(
            max_batch_size=max_batch_size_cfg, 
            max_seq_length=max_seq_length, 
            dtype=model.tok_embeddings_extended.weight.dtype
        )
    # Create output tensor
    seq = torch.empty((batch_size, total_tokens), dtype=torch.int, device=device)
    
    # Fill in prefix tokens
    seq[:, :prefix_len] = prefix_tokens
    
    # Prefill with conditioning token and prefix tokens
    next_token, next_pos = prefill_with_prefix(
        model, cond_combined, prefix_combined, cfg_scale, padding_mask=padding_mask, lens=lens, **sampling_kwargs
    )

    # Store first generated token
    seq[:, prefix_len:prefix_len+1] = next_token

    # Collect log-probs for GRPO if requested
    gen_logps_list = []

    # Generate remaining tokens
    if num_new_tokens > 1:
        input_pos = torch.tensor([next_pos], device=device, dtype=torch.int)
        generated_tokens, _ = decode_n_tokens(
            model, next_token, input_pos, num_new_tokens - 1,
            cfg_scale, cfg_interval, padding_mask=padding_mask, lens=lens, **sampling_kwargs
        )
        seq[:, prefix_len+1:] = torch.cat(generated_tokens, dim=1)

    if return_logps:
        # Recompute log-probs by replaying generation through fresh KV caches.
        # This avoids modifying prefill/decode functions to return logits.
        with torch.device(device):
            max_batch_size_cfg = batch_size * 2 if cfg_scale > 1.0 else batch_size
            model.setup_caches(
                max_batch_size=max_batch_size_cfg,
                max_seq_length=max_seq_length,
                dtype=model.tok_embeddings_extended.weight.dtype
            )

        # Re-run prefill
        input_pos_cond_prefix = torch.arange(0, 1 + prefix_len, device=device, dtype=torch.int)
        logits_prefill, _ = model(prefix_combined, cond_idx=cond_combined, input_pos=input_pos_cond_prefix, non_mask=padding_mask, lens=lens)

        # Log-prob of first generated token
        temperature = sampling_kwargs.get('temperature', 1.0)
        first_logits = logits_prefill[:, -1, :] / max(temperature, 1e-5)
        first_log_probs = F.log_softmax(first_logits, dim=-1)
        gen_logps_list.append(first_log_probs.gather(-1, next_token).squeeze(-1))

        # Now decode remaining tokens and collect log-probs
        if num_new_tokens > 1:
            cur_token = next_token
            input_pos_lp = torch.tensor([next_pos], device=device, dtype=torch.int)
            for i in range(num_new_tokens - 1):
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    logits_step, _ = model(cur_token.view(-1, 1), cond_idx=None, input_pos=input_pos_lp)
                step_logits = logits_step[:, -1, :] / max(temperature, 1e-5)
                step_log_probs = F.log_softmax(step_logits, dim=-1)
                actual_token = seq[:, prefix_len + 1 + i:prefix_len + 2 + i]
                gen_logps_list.append(step_log_probs.gather(-1, actual_token).squeeze(-1))
                cur_token = actual_token
                input_pos_lp += 1

        gen_logps = torch.stack(gen_logps_list, dim=1)  # (B, num_new_tokens)
        return seq, gen_logps

    return seq

### Visualization functions #### 
def decode_tokens_to_images_batched(tokens, vq_model, n_channels, tokens_per_channel, 
                            latent_size, codebook_embed_dim):
    """
    Decode token sequences to images.
    
    Args:
        tokens: (B, seq_len) token sequence with SOC/EOC markers
                Format: [SOC_0, patches_0..., EOC_0, SOC_1, patches_1..., EOC_1, ..., EOS]
        vq_model: VQ decoder model
        n_channels: Number of channels in the sequence
        tokens_per_channel: Tokens per channel (258 = 1 SOC + 256 patches + 1 EOC)
        vocab_size: VQ codebook size
        latent_size: Latent spatial size (e.g., 16)
        codebook_embed_dim: Codebook embedding dimension
        
    Returns:
        decoded_images: (B, n_channels, 3, H, W) decoded images
    """
    batch_size = tokens.shape[0]
    
    # Make sure tokens length matches expected length
    # If token length = expected_length + 1, likely because EOS token was not removed from tokens
    assert tokens.shape[1] == n_channels * tokens_per_channel, "Tokens length (={}) does not match expected length (={})".format(tokens.shape[1], n_channels * tokens_per_channel)
    
    # Reshape: [B, n_channels, tokens_per_channel]
    reshaped = tokens.reshape(batch_size, n_channels, tokens_per_channel)
    
    # Extract only patch tokens (skip SOC at index 0, EOC at index -1)
    patch_tokens = reshaped[:, :, 1:-1]  # [B, n_channels, patches_per_channel]

    ## TEMPORARY: hard-coded fix to replace patch tokens outside of image vocabulary with
    ## TODO: deal with this in decode_one_token function to only sample from logits from image vocabulary within patch token positions ?
    patch_tokens = patch_tokens.clamp(0, vq_model.quantize.n_e - 1)

    # assert patch tokens are within image vocabulary dimension (codebook size)
    assert patch_tokens.max() < vq_model.quantize.n_e, "Patch tokens are out of image vocabulary dimension"
    
    # Decode all channels in one batched pass
    # Reshape from [B, n_channels, patches] to [B*n_channels, patches]
    flat_tokens = patch_tokens.reshape(batch_size * n_channels, -1)
    qzshape = [batch_size * n_channels, codebook_embed_dim, latent_size, latent_size]
    flat_images = vq_model.decode_code(flat_tokens, qzshape)  # [B*n_channels, 3, H, W]

    # Reshape back to [B, n_channels, 3, H, W]
    return flat_images.reshape(batch_size, n_channels, *flat_images.shape[1:])

def decode_tokens_to_images(tokens, vq_model, n_channels, tokens_per_channel, 
                            latent_size, codebook_embed_dim):
    """
    Decode token sequences to images.
    
    Args:
        tokens: (B, seq_len) token sequence with SOC/EOC markers
                Format: [SOC_0, patches_0..., EOC_0, SOC_1, patches_1..., EOC_1, ..., EOS]
        vq_model: VQ decoder model
        n_channels: Number of channels in the sequence
        tokens_per_channel: Tokens per channel (258 = 1 SOC + 256 patches + 1 EOC)
        vocab_size: VQ codebook size
        latent_size: Latent spatial size (e.g., 16)
        codebook_embed_dim: Codebook embedding dimension
        
    Returns:
        decoded_images: (B, n_channels, 3, H, W) decoded images
    """
    batch_size = tokens.shape[0]
    
    # Make sure tokens length matches expected length
    # If token length = expected_length + 1, likely because EOS token was not removed from tokens
    assert tokens.shape[1] == n_channels * tokens_per_channel, "Tokens length (={}) does not match expected length (={})".format(tokens.shape[1], n_channels * tokens_per_channel)
    
    # Reshape: [B, n_channels, tokens_per_channel]
    reshaped = tokens.reshape(batch_size, n_channels, tokens_per_channel)
    
    # Extract only patch tokens (skip SOC at index 0, EOC at index -1)
    patch_tokens = reshaped[:, :, 1:-1]  # [B, n_channels, patches_per_channel]

    ## TEMPORARY: hard-coded fix to replace patch tokens outside of image vocabulary with
    ## TODO: deal with this in decode_one_token function to only sample from logits from image vocabulary within patch token positions ?
    patch_tokens = patch_tokens.clamp(0, vq_model.quantize.n_e - 1)

    # assert patch tokens are within image vocabulary dimension (codebook size)
    assert patch_tokens.max() < vq_model.quantize.n_e, "Patch tokens are out of image vocabulary dimension"
    
    # Decode each channel
    decoded_channels = []
    for ch in range(n_channels):
        ch_tokens = patch_tokens[:, ch, :]
        qzshape = [batch_size, codebook_embed_dim, latent_size, latent_size]
        ch_images = vq_model.decode_code(ch_tokens, qzshape)
        decoded_channels.append(ch_images)
    
    return torch.stack(decoded_channels, dim=1)  # [B, n_channels, 3, H, W]

def load_and_prepare_prefixes(
    codes: np.ndarray,
    labels: np.ndarray,
    n_prefix_channels: int,
    tokens_per_channel: int = 258,
    aug_idx: int = 0,
    cfg_scale: float = 1.0,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Load and prepare a batch of prefix tokens and labels.
    
    Args:
        code_dir: Path to code directory
        label_dir: Path to label directory
        n_prefix_channels: Number of prefix channels to extract
        batch_indices: List of file indices to load
        tokens_per_channel: Tokens per channel (default 258)
        aug_idx: Augmentation index to use
        device: Device to move tensors to
        
    Returns:
        prefix_tokens: (B, prefix_len) tensor of prefix tokens
        cond_idx: (B, 1) tensor of conditioning labels
        filenames: List of filenames
    """
    
    prefixes = []
    prefix_labels = []
    
    for code, label in zip(codes, labels):
        # print(code.shape)
        if len(code.shape) > 1 and code.shape[0] > 1:
            code = code[aug_idx]
        elif len(code.shape) > 1:
            code = code[0]
        
        # Extract prefix
        prefix = extract_prefix_channels(code, n_prefix_channels, tokens_per_channel)
        prefixes.append(prefix)

        prefix_labels.append(label)
    
    # Stack into tensors
    prefix_tokens = torch.from_numpy(np.stack(prefixes)).long()
    cond_idx = torch.from_numpy(np.stack(prefix_labels)).long()
    
    # Ensure cond_idx has shape (B, 1)
    if cond_idx.dim() == 1:
        cond_idx = cond_idx.unsqueeze(1)
    
    if device is not None:
        prefix_tokens = prefix_tokens.to(device)
        cond_idx = cond_idx.to(device)
    
    return prefix_tokens, cond_idx

#################################################################################
#              Validation Code Loading and Prefix-Based Generation              #
#################################################################################

def load_validation_codes(
    code_dir: str,
    label_dir: str,
    num_samples: Optional[int] = None,
    aug_idx: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Load pre-tokenized validation codes and labels from .npy files.
    
    Args:
        code_dir: Path to directory containing code .npy files
        label_dir: Path to directory containing label .npy files  
        num_samples: Number of samples to load (None for all)
        aug_idx: Augmentation index to select (0 for first augmentation)
        
    Returns:
        codes: List of code arrays, each shape (seq_len,)
        labels: List of label arrays
        filenames: List of filenames (without extension)
    """
    code_files = sorted([f for f in os.listdir(code_dir) if f.endswith('.npy')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])
    
    if num_samples is not None:
        code_files = code_files[:num_samples]
        label_files = label_files[:num_samples]
    
    codes = []
    labels = []
    filenames = []
    
    for code_file, label_file in zip(code_files, label_files):
        # Load code: shape (num_aug, seq_len) or (seq_len,)
        code = np.load(os.path.join(code_dir, code_file))
        if len(code.shape) > 1 and code.shape[0] > 1:
            code = code[aug_idx]
        elif len(code.shape) > 1:
            code = code[0]
        
        # Load label
        label = np.load(os.path.join(label_dir, label_file))
        
        codes.append(code)
        labels.append(label)
        filenames.append(code_file.replace('.npy', ''))
    
    return codes, labels, filenames

def extract_prefix_channels(
    code: np.ndarray,
    n_prefix_channels: int,
    tokens_per_channel: int = 258,
) -> np.ndarray:
    """
    Extract prefix channels from a full code sequence.
    
    Args:
        code: Full code sequence shape (seq_len,) containing all channels
              Format: [SOC_0, patches_0..., EOC_0, SOC_1, patches_1..., EOC_1, ...]
        n_prefix_channels: Number of channels to extract as prefix
        tokens_per_channel: Tokens per channel (1 SOC + 256 patches + 1 EOC = 258)
        
    Returns:
        prefix: Prefix tokens shape (n_prefix_channels * tokens_per_channel,)
    """
    prefix_len = n_prefix_channels * tokens_per_channel
    return code[:prefix_len]

@torch.no_grad()
def sample_with_prefix(
    model,
    code_dir: str,
    label_dir: str,
    n_prefix_channels: int,
    n_total_channels: int,
    batch_size: int = 4,
    num_samples: Optional[int] = None,
    tokens_per_channel: int = 258,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    cfg_scale: float = 1.0,
    cfg_interval: int = -1,
    device: torch.device = None,
    aug_idx: int = 0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """
    Main function to sample from GPT-CA model using prefix conditioning.
    
    Loads validation codes, extracts prefix channels, and generates remaining channels.
    
    Args:
        model: Channel-adaptive GPT model
        code_dir: Path to validation codes directory
        label_dir: Path to validation labels directory
        n_prefix_channels: Number of channels to use as prefix/conditioning
        n_total_channels: Total number of channels to generate
        batch_size: Batch size for generation
        num_samples: Total number of samples to generate (None for all)
        tokens_per_channel: Tokens per channel (default 258)
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        cfg_scale: Classifier-free guidance scale
        cfg_interval: CFG interval
        device: Device for computation
        aug_idx: Augmentation index to use from codes
        
    Returns:
        all_generated: List of generated token sequences (B, total_tokens)
        all_ground_truth: List of ground truth full sequences
        all_filenames: List of all filenames processed
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Get list of files
    code_files = sorted([f for f in os.listdir(code_dir) if f.endswith('.npy')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])

    # create tuple of code_files and label_files
    code_label_tuples = list(zip(code_files, label_files))
    # shuffle the tuple
    random.shuffle(code_label_tuples)
    # unzip the tuple
    code_files, label_files = zip(*code_label_tuples)
    
    total_files = len(code_files)
    if num_samples is not None:
        total_files = min(num_samples, total_files)
    
    all_generated = []
    all_ground_truth = []
    all_filenames = []
    
    # Process in batches
    for start_idx in range(0, total_files, batch_size):
        end_idx = min(start_idx + batch_size, total_files)
        batch_indices = list(range(start_idx, end_idx))

        codes = np.array([np.load(os.path.join(code_dir, code_files[idx])) for idx in batch_indices])
        labels = np.array([np.load(os.path.join(label_dir, label_files[idx])) for idx in batch_indices])
        
        # print(labels.shape)
        # Load and prepare batch
        prefix_tokens, true_labels = load_and_prepare_prefixes(
            codes, labels, n_prefix_channels,
            tokens_per_channel, aug_idx, device
        )

        # print(true_labels.shape)
        # print("true_labels:")
        # print(true_labels)
        # print("labels:")
        # print(labels)

        # print(labels)
        cond_idx_batch = torch.tensor(labels, dtype=torch.bfloat16).unsqueeze(1).to(device)
        # print(cond_idx_batch.shape)
        # print(cond_idx_batch)
        
        # Load ground truth (full sequences)
        ground_truth = []
        for idx in batch_indices:
            code = np.load(os.path.join(code_dir, code_files[idx]))
            if len(code.shape) > 1 and code.shape[0] > 1:
                code = code[aug_idx]
            elif len(code.shape) > 1:
                code = code[0]
            ground_truth.append(torch.from_numpy(code).long())
        ground_truth = torch.stack(ground_truth).to(device)
        
        # Generate
        generated = generate_with_prefix(
            model=model,
            cond_idx=cond_idx_batch,
            prefix_tokens=prefix_tokens,
            n_total_channels=n_total_channels,
            tokens_per_channel=tokens_per_channel,
            cfg_scale=cfg_scale,
            cfg_interval=cfg_interval,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_logits=True,
        )
        
        all_generated.append(generated)
        all_ground_truth.append(ground_truth)
        all_filenames.extend(code_files[start_idx:end_idx])
    
    return all_generated, all_ground_truth, all_filenames
