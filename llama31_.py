import os
import glob
import fire
import time
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

#from tokenizer import Tokenizer

# -------------------------------------MODEL ARGUMENTS ----------------------------------------
@dataclass
class ModelArgs:
    dim: int = 4096 #distributed repres of the tokens 
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 #128k
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False
    max_batch_size: int = 32
    max_seq_len: int = 2048 
    flash: bool = False # use flash attention?
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            if hasattr(self, k):
                setattr(self,k, v)
            if self.n_kv_heads == None:
                self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0, f'Use NICE numbers for your heads!'
        assert self.dim % self.n_heads == 0, f'Make sure to use dividable num_embed and num_heads!'

# ----------------------------------Rotatary Positional Embeding----------------------------------
def apply_scaling(freqs: torch.Tensor):
    # RoPE scaling (values obtained from grid search)
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled: #for long context len than what its trained on!
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    freqs_cis_real = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return freqs_cis_real

def apply_rotary_emb(x, freqs_cis):
    # shape gymnastics let's go
    # x is (B, T, n_heads, head_dim), e.g. (4, 8, 32, 128)
    # freqs_cis is (T, head_dim/2, 2), e.g. (8, 64, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    # xshaped is (B, T, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    # freqs_cis becomes (1, T, 1, head_dim/2, 2), e.g. (1, 8, 1, 64, 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    # x_out2 at this point is (B, T, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)
    x_out2 = x_out2.flatten(3)
    # x_out2 is now (B, T, n_heads, head_dim), e.g. (4, 8, 32, 128)
    return x_out2.type_as(x)

# ----------------------------------Main RMSNorm---------------------------------------------------
class RMSNorm(nn.Module):
    """
    claim is re-scaling is the key not re-centering, so get rid of the mean n only
    re-scale the varience; across columns: weights
    """
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
    def forward(self, x):
        #Returns this tensor cast to the type of the given tensor. Equivalent to self.type(tensor.type())
        out = self._norm(float(x)).type_as(x)
        return out * self.weights

# ----------------------------------for GQA?------------------------------------------------------
def repeat_kv(x, n_rep):
    (bsize, Tseqlen, n_kv_heads, h_dim) = x.shape
    if n_rep ==1:#regular multi-head attention
        return x 
    return (x[:,:,None,:].expand(bsize, Tseqlen, n_rep, h_dim).reshape(bsize, Tseqlen, n_kv_heads * n_rep , h_dim))

# ----------------------------------KV CACHE------------------------------------------------------  
class KVCache(nn.Module):
    def __init__(self, batch_size, seq_len, n_kv_heads, head_dim, dtype, device):
        super().__init__()
        cache_shape = (batch_size, seq_len, n_kv_heads, head_dim) #(B,T,nh, hs)
        self.register_buffer('cache_k', torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer('cache_v', torch.zeros(cache_shape, dtype=dtype, device=device))
        
    def update(self, start_pos, xk, xv):
        seqlen = xk.size(1)
        self.cache_k[:, start_pos: start_pos + seqlen] = xk
        self.cache_v[:, start_pos: start_pos + seqlen] = xv
        xk = self.cache_k[:, :  start_pos + seqlen] # (B,T,nh, hs)
        xv = self.cache_v[:, :  start_pos + seqlen] # (B,T,nh, hs)
        return xk, xv

# ----------------------------------attention---------------------------------------------------
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.flash = args.flash
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads #for GQA
        model_parallize = 1 #for 1 gpu 
        self.n_local_heads = args.n_heads //  model_parallize #heads per GPU
        self.n_local_kv_head = self.kv_heads // model_parallize
        self.n_rep = self.n_local_heads // self.n_local_kv_head # for repeat_cache for GQA
        self.head_dim = args.dim // args.n_heads #the embed dim each head in ech GPU will recieve!
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False) # (T, )
        self.wk = nn.linear(args.dim, self.n_kv_heads * self.head_dim, bias=False) # to support Grouped Query 
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        # for KV Cache 
        self.cache = None
        
    def forward(self, x, start_pos, freqs_cis, mask):
        B, T, C = x.shape #(batch_size, seq_len, dim)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(B, T, self.n_local_heads, self.head_dim) 
        xk = xk.view(B, T, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, T, self.n_kv_heads, self.head_dim)
        
        #apply Rotatory Embeding (Positional embed); rotate query, keys (RoPE)
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        
        # KV cache update
        if self.cache is not None:
            # update the KV cache with current KV and get all the previous KVs
            xk, xv = self.cache.update(start_pos, xk, xv)
        # repeat k/v heads if n_kv_heads < n_heads (GQA)
        xk = repeat_kv(xk, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        # make heads be a batch dim
        xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv)) #(B, T, nh, hs) --T--> (B, nh, T, hs)
        # attention
        if self.flash:
            output = F.scaled_dot_product_attention(xq, xk, xv, mask)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
        # concatenate all the heads
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        # output projection
        proj = self.wo(output)
        return proj     
# ----------------------------------FFN---------------------------------------------------
class FeedForward(nn.Module): #acts like a map k:v where k is the input text n v is the distribution over the output vocab!
    """Because reaching a given parameter count (e.g. 2B) with a given embedding size (e.g. 2048) you need 
    to pick between lots of blocks (deeper network) with "lighter" mlps, or fewer blocks with wider MLPs.
    The advantage of fewer layers with small(-ish) embedding size is the less memory is spent for kv (context) cache,
    and less compute spent on attention for long contexts.
    """
    def __init__(
        self,
        dim,
        hidden_dim,
        multiple_of,
        ffn_dim_multiplier):
        
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3) #ffn 2/3 network
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of) #to make it NICE num
        self.w1 = nn.Linear(dim, hidden_dim, bias=False) #project_gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) #project_up
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) #project_down
        
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
# ----------------------------------Block---------------------------------------------------
class Block(nn.Module):
    def __inint__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        
        self.mlp = FeedForward(
            dim = args.dim,
            hidden_dim= 4 * args.hidden_dim,
            multiple_of = args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplie)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.mlp_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x:torch.Tensor,  start_pos, freqs_cis, mask: Optional[torch.Tensor],):
        h = x + self.attention(self.attention_norm(x, start_pos, freqs_cis, mask))
        out = h + self.mlp(self.mlp_norm(x))
        return out
# ----------------------------------Main TF---------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs): 
        super().__init__()
        self.params = params 
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        #main 
        self.token_embed = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList(list(Block(params) for _ in range(params.n_layers)))
        self.norm = RMSNorm(params.dim)
        self.lm_head = nn.Linear(params.dim, params.vocab_size, bias=False)
        
        self.freqs_cis = precompute_freqs_cis(
            dim = params.dim // params.n_heads,
            end= params.max_seq_len * 2,
            theta= params.rope_theta,
            use_scaled=params.use_scaled_rope)
        
    def forward(): #logits + loss
        pass
    
    def forward_inference():
        pass
    
    def config_optimizer():
        pass