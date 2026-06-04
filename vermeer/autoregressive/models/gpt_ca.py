# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from dataclasses import dataclass
from typing import Optional, List, Union


import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.drop_path import DropPath

import numpy as np
import scipy.ndimage

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02

    # Channel-adaptive settings
    n_max_channels: int = 4
    block_size_per_channel: int = 256  # patches per channel (e.g., 16x16=256)
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0
    label_smooth: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'ca'

    vocab_size: int = 16384  # base vocab for image patches # TODO: pad extended vocab size to multiple of 8 for tensor cores?
    cls_token_num: int = 1
    max_batch_size: int = 32
    # max_seq_len: int = 2048

    pretrained_gpt: bool = False
    
    # TODO: check if SOS and EOS tokens are handled correctly in the code, tokenizer, and training 
    @property
    def extended_vocab_size(self) -> int:
        """Extended vocab: base + SOC_i + EOC_i + EOS"""
        return self.vocab_size + 2 * self.n_max_channels + 1
    
    @property 
    def tokens_per_channel(self) -> int:
        """SOC + patches + EOC per channel"""
        return 1 + self.block_size_per_channel + 1
    
    @property
    def max_seq_length(self) -> int:
        """n_channels * (SOC + patches + EOC) + EOS"""
        return self.n_max_channels * self.tokens_per_channel + 1
    
    # TODO: check indexing is correct (channel_idx should be 0-indexed)
    def soc_token_id(self, channel_idx: int) -> int:
        """Get SOC token ID for given channel"""
        return self.vocab_size + channel_idx
    
    def eoc_token_id(self, channel_idx: int) -> int:
        """Get EOC token ID for given channel"""
        return self.vocab_size + self.n_max_channels + channel_idx
    
    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID"""
        return self.vocab_size + 2 * self.n_max_channels


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

#################################################################################
#                      Embedding Layers for full ESM embedding                  #
#################################################################################

## from DeepLoc 2.0: https://github.com/teevee112/DeepLoc-2.0/blob/main/src/attr_prior.py
def smooth_tensor_1d(input_tensor, smooth_sigma):
    """
    Smooths an input tensor along a dimension using a Gaussian filter.
    Arguments:
        `input_tensor`: a A x B tensor to smooth along the second dimension
        `smooth_sigma`: width of the Gaussian to use for smoothing; this is the
            standard deviation of the Gaussian to use, and the Gaussian will be
            truncated after 1 sigma (i.e. the smoothing window is
            1 + (2 * sigma); sigma of 0 means no smoothing
    Returns an array the same shape as the input tensor, with the dimension of
    `B` smoothed.
    """
    # Generate the kernel
    if smooth_sigma == 0:
        sigma, truncate = 1, 0
    else:
        sigma, truncate = smooth_sigma, 1
    base = np.zeros(1 + (2 * sigma))
    base[sigma] = 1  # Center of window is 1 everywhere else is 0
    kernel = scipy.ndimage.gaussian_filter(base, sigma=sigma, truncate=truncate)
    # Match kernel dtype to input tensor dtype (instead of float32)
    kernel = torch.tensor(kernel, dtype=input_tensor.dtype, device=input_tensor.device)

    # Expand the input and kernel to 3D, with channels of 1
    input_tensor = torch.unsqueeze(input_tensor, dim=1)
    # kernel = torch.unsqueeze(torch.unsqueeze(kernel, dim=0), dim=1).float()
    kernel = torch.unsqueeze(torch.unsqueeze(kernel, dim=0), dim=1)
    padded_input = F.pad(input_tensor, (sigma,sigma),"replicate")
    smoothed = torch.nn.functional.conv1d(
        padded_input, kernel
    )
    return torch.squeeze(smoothed, dim=1)

## from DeepLoc 2.0: https://github.com/teevee112/DeepLoc-2.0/blob/main/src/model.py
## note: uses 1 attention head in practice
class PLMAttentionHead(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super(PLMAttentionHead, self).__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.preattn_ln = nn.LayerNorm(hidden_dim//n_heads)
        self.Q = nn.Linear(hidden_dim//n_heads, n_heads, bias=False)
        torch.nn.init.normal_(self.Q.weight, mean=0.0, std=1/(hidden_dim//n_heads))

    def forward(self, x, np_mask, lengths):
        # input (batch, seq_len, embed)
        n_heads = self.n_heads
        hidden_dim = self.hidden_dim
        x = x.view(x.size(0), x.size(1), n_heads, hidden_dim//n_heads)
        x = self.preattn_ln(x)
        mul = (x * \
            self.Q.weight.view(1, 1, n_heads, hidden_dim//n_heads)).sum(-1) \
            #* np.sqrt(5)
            #/ np.sqrt(hidden_dim//n_heads)
        mul_score_list = []
        for i in range(mul.size(0)):
            # (1, L) -> (1, 1, L) -> (1, L) -> (1, L, 1)
            mul_score_list.append(F.pad(smooth_tensor_1d(mul[i, :lengths[i], 0].unsqueeze(0), 2).unsqueeze(0),(0, mul.size(1)-lengths[i]),"constant").squeeze(0))
        
        mul = torch.cat(mul_score_list, dim=0).unsqueeze(-1)
        mul = mul.masked_fill(~np_mask.unsqueeze(-1), float("-inf"))
        
        attns = F.softmax(mul, dim=1) # (b, l, nh)
        x = (x * attns.unsqueeze(-1)).sum(1)
        x = x.view(x.size(0), -1)
        return x, attns.squeeze(2)

# attention pooling 
class ESMAttentionPoolEmbedder(nn.Module):
    """
    Embeds full ESM embedding into vector representations using attention pooling.
    Also handles label dropout for classifier-free guidance. (skipped for now)
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()

        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

        #TODO: this should be extraneous b/c final layer of esm has LayerNorm. but double-check 
        #TODO: without self.lin, redundant because of preattn_ln in PLMAttentionHead
        # self.initial_ln = nn.LayerNorm(in_channels) 
        # self.lin = nn.Linear(in_channels, hidden_size)
        self.attn_head = PLMAttentionHead(in_channels, 1) # input -> attention (single head) -> projection mlp 

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, non_mask, lens, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption, attns = self.attn_head(caption, non_mask, lens)
        embeddings = self.cap_proj(caption).unsqueeze(1)
        return embeddings

#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    """
    Channel-Adaptive GPT for multi-channel microscopy images.
    
    Token sequence: [SOS, SOC_0, patches_ch0..., EOC_0, SOC_1, patches_ch1..., EOC_1, ..., EOS]
    Each token gets: token_embed + 2D_RoPE + channel_num_embed
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.extended_vocab_size = config.extended_vocab_size
        self.n_layer = config.n_layer
        self.n_max_channels = config.n_max_channels
        self.block_size_per_channel = config.block_size_per_channel
        self.tokens_per_channel = config.tokens_per_channel
        self.pretrained_gpt = config.pretrained_gpt
        self.cls_token_num = config.cls_token_num
        self.model_type = config.model_type
        if self.model_type == 'ca':
            # SOS embedding (injected, not predicted)
            self.sos_embed = nn.Parameter(torch.zeros(1, config.dim))
            nn.init.trunc_normal_(self.sos_embed, std=config.initializer_range)
        elif self.model_type == 'ca_binary_prefix':
            self.cls_embedding = LabelEmbedder(2, config.dim, config.class_dropout_prob)
        elif self.model_type == 'ca_esm_embed_mean_pool':
            esm_dim = 1152
            self.cls_embedding = CaptionEmbedder(esm_dim, config.dim, config.class_dropout_prob, token_num=self.cls_token_num)
        elif self.model_type == 'ca_esm_embed_full':
            esm_dim = 1152
            # input esm is [AA, 1152]
            # self.cls_embedding = ESMConvPoolEmbedder(esm_dim, config.dim, config.class_dropout_prob, token_num=self.cls_token_num)
            self.cls_embedding = ESMAttentionPoolEmbedder(esm_dim, config.dim, config.class_dropout_prob, token_num=self.cls_token_num)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Token embeddings (base vocab: patches) (only used for loading in weights from pretrained checkpoint)
        if self.pretrained_gpt:
            self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim).requires_grad_(False)
        # Token embeddings (extended vocab: patches + SOC_i + EOC_i + EOS)
        self.tok_embeddings_extended = nn.Embedding(config.extended_vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        self.label_smooth = config.label_smooth
        
        # Channel number embedding (similar to GPT segment embedding, to distinguish different microscopy stains)
        # TODO: change nomenclature (replace "channel" with "stain"?)
        # channels are 1..n_max_channels
        # self.channel_num_embed = nn.Embedding(config.n_max_channels, config.dim)
        # nn.init.trunc_normal_(self.channel_num_embed.weight.data, std=config.initializer_range)
        
        # skip this for now b/c might interefere w RoPE
        # # 1D positional embedding (absolute position in sequence)
        # self.pos_1LC = nn.Parameter(torch.zeros(1, config.max_seq_length, config.dim))
        # nn.init.trunc_normal_(self.pos_1LC, std=config.initializer_range)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))
        
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        # Output layer (predicts base vocab, only used for loading in weights from pretrained checkpoint)
        if self.pretrained_gpt:
            self.output = nn.Linear(config.dim, config.vocab_size, bias=False).requires_grad_(False)
        # Output layer (predicts extended vocab)
        self.output_extended = nn.Linear(config.dim, config.extended_vocab_size, bias=False)

        # 2D RoPE for multi-channel (resets per channel)
        self.grid_size = int(config.block_size_per_channel ** 0.5)
        assert self.grid_size * self.grid_size == config.block_size_per_channel
        self.freqs_cis = precompute_freqs_cis_2d_multi_channel(
            self.grid_size, config.dim // config.n_head, config.n_max_channels, config.rope_base
        )
        
        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output_extended.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def extend_head(self):
        """Copies weights from the base head (e.g. from pretrained checkpoint) to the extended head."""
        assert self.pretrained_gpt, "only extend head if pretrained gpt is being loaded"
        with torch.no_grad():
            self.output_extended.weight[:self.vocab_size] = self.output.weight

    def extend_tok_embeddings(self):
        """Copies weights from the base tok_embeddings to the extended tok_embeddings."""
        assert self.pretrained_gpt, "only extend tok_embeddings if pretrained gpt is being loaded"
        with torch.no_grad():
            self.tok_embeddings_extended.weight[:self.vocab_size] = self.tok_embeddings.weight

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        
        self.freqs_cis = precompute_freqs_cis_2d_multi_channel(
            self.grid_size, self.config.dim // self.config.n_head, 
            self.config.n_max_channels, self.config.rope_base
        )

    def get_channel_indices(self, n_channels: int, device: torch.device) -> torch.Tensor:
        """
        Build channel index tensor for segment embedding.
        Returns indices: 1..n_channels for channel tokens
        """
        indices = []
        for ch in range(n_channels):
            # SOC, patches, EOC all get channel index (ch + 1)
            indices.extend([ch + 1] * self.tokens_per_channel)
        return torch.tensor(indices, device=device)

    def forward(
        self, 
        idx: torch.Tensor,
        cond_idx: torch.Tensor, # cond_idx_or_embed (prefix_condition or [0] in unconditional case) 
        input_pos: Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        non_mask: Optional[torch.Tensor] = None,
        lens: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for channel-adaptive GPT.
        
        Supports
        1. Training: prepends SOS token (input_pos is None)
        2. Inference: decode_n_tokens (input_pos is not None)
        
        Args:
            idx: (B, seq_len) token indices
            input_pos: position indices for inference (None for training)
            targets: (B, seq_len) target tokens for loss computation
            mask: optional attention mask
            valid: optional validity mask for loss
        """

        if cond_idx is not None:
            batch_size = cond_idx.shape[0]
            
            # If unconditional, use sos_embed
            if self.model_type == 'ca':
                cond_embeddings = self.sos_embed.unsqueeze(1).expand(batch_size, -1, -1).contiguous()
            elif self.model_type == 'ca_binary_prefix':
                cond_idx = cond_idx.squeeze(1) ## TODO: fix this in dataloader / extract_codes_ca.py instead 
                # print(f"cond_idx shape: {cond_idx.shape}")
                cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
                # print(f"cond_embeddings shape: {cond_embeddings.shape}")
            elif self.model_type == 'ca_esm_embed_mean_pool' or self.model_type == 'ca_esm_embed_full':
                ## TODO: fix this in dataloader / extract_codes_ca.py instead 
                # print("cond_idx shape:", cond_idx.shape) # in preprocessing, cls and eos token are removed, mean pooled\
                # print("cond_idx:", cond_idx)
                # print(self.cls_token_num)
                # look for all 0s or nans in cond_idx
                if torch.isnan(cond_idx).any():
                    raise ValueError("cond_idx contains nans")
                # Check if any batch element has all zeros across embedding dimension
                # #TODO: double check this
                # if self.model_type == 'ca_esm_embed_mean_pool':
                #     all_zeros_mask = (cond_idx == 0).all(dim=-1)  # (B,)
                # elif self.model_type == 'ca_esm_embed_full':
                #     all_zeros_mask = (cond_idx.sum(dim=1) == 0).all(dim=-1)  # (B,)
                # if all_zeros_mask.any():
                #     raise ValueError("cond_idx contains all zeros")
                if self.model_type == 'ca_esm_embed_full':
                    cond_embeddings = self.cls_embedding(cond_idx, train=self.training, non_mask=non_mask, lens=lens)[:,:self.cls_token_num]
                    
                else:
                    # TODO: add unsqueeze(1) to cond_idx in dataloader
                    cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
                    # print("cond_embeddings shape:", cond_embeddings.shape)
                # print("cond_embeddings shape:", cond_embeddings.shape)
            else: # conditional case
                raise NotImplementedError("Conditional generation is not implemented for CA model")

        if idx is not None and cond_idx is not None: # training or naive inference
            token_embeddings = self.tok_embeddings_extended(idx)
            # # Add channel number embeddings
            # channel_indices = self.get_channel_indices(self.n_max_channels, idx.device)
            # channel_indices = channel_indices[:idx.shape[1]].unsqueeze(0).expand(idx.shape[0], -1)
            # channel_embeddings = self.channel_num_embed(channel_indices)
            # token_embeddings = token_embeddings + channel_embeddings
            # print(f"cond_embeddings shape: {cond_embeddings.shape}")
            # print(f"token_embeddings shape: {token_embeddings.shape}")
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings) # TODO: confirm if correct to apply dropout on cond_embeddings? 
            self.freqs_cis = self.freqs_cis.to(h.device)
        else:
            if cond_idx is not None: # prefill in inference
                token_embeddings = cond_embeddings
            else: # decode_n_tokens(kv cache) in inference
                token_embeddings = self.tok_embeddings_extended(idx)
            
            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis
        
        # if self.training:
        if input_pos is None:
            freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]
        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)
        
        # output layers
        h = self.norm(h)
        logits = self.output_extended(h).float()
        
        if self.training:
            logits = logits[:, self.cls_token_num - 1:].contiguous()

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none', label_smoothing=self.label_smooth)
            valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), label_smoothing=self.label_smooth)

        return logits, loss

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)



#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache

# TODO: check if RoPE is implemented correctly for multi-channel sequences (e.g. only do RopE on patch tokens, not special tokens?)
def precompute_freqs_cis_2d_multi_channel(
    grid_size: int, n_elem: int, n_channels: int, base: int = 10000
):
    """
    Precompute 2D RoPE frequencies for multi-channel sequences.
    
    Sequence structure per channel: [SOC (zero), patches (2D RoPE), EOC (zero)]
    Full sequence: [SOS (zero), ch0_tokens, ch1_tokens, ...]
    
    Args:
        grid_size: sqrt of patches per channel (e.g., 16 for 256 patches)
        n_elem: head dimension
        n_channels: number of channels
        base: RoPE base frequency
    
    Returns:
        freqs_cis: (1 + n_channels*(1+grid_size^2+1) + 1, n_elem//2, 2)
    """
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1)
    patches_cache = cache_grid.flatten(0, 1)  # (grid_size^2, head_dim//2, 2)
    
    zero = torch.zeros(1, n_elem // 2, 2)
    
    # Build per-channel cache: [SOC (zero), patches (2D), EOC (zero)]
    channel_cache = torch.cat([zero, patches_cache, zero], dim=0)
    
    # Build full sequence: [SOS (zero), channels...,]
    all_caches = [zero]  # SOS
    for _ in range(n_channels):
        all_caches.append(channel_cache)
    
    return torch.cat(all_caches, dim=0) 


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)



#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        

GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 
}