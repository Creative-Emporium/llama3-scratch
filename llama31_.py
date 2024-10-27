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

from tokenizer import Tokenizer

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
# ----------------------------------attention---------------------------------------------------
class Attention(nn.Module):
    pass

# ----------------------------------FFN---------------------------------------------------
class FeedForward(nn.module): #acts like a map k:v where k is the input text n v is the distribution over the output vocab!
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
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
# ----------------------------------Block---------------------------------------------------
class Block(nn.module):
    def __inint__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.mlp = FeedForward()
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.mlp_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x:torch.Tensor,  start_pos: int,freqs_cis: torch.Tensor, mask: Optional[torch.Tensor],):
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
        #TODO: ADD self.freqs_cis   

    def forward(): #logits + loss
        pass
    
    def forward_inference():
        pass
    
    def config_optimizer():
        pass