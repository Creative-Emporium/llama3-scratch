#TODO: 1)add split to the dataloader 2) 

#model.embed_tokens.weight torch.Size([128256, 4096]) #4096 is the num_embed of the mode
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.functional as F 
#==================== let the fun begins ======================================
#==================== Model Arguments ======================================
@dataclass
class llamaConfig: #model args 
    block_size : int = None #T dimension# it varries for llama
    dim: int = 4096 #num_embed
    n_layers: int = 32 # 
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 #128k for 1B 
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000 #in paper; chose 500k to better support longer contexts;
    use_scaled_rope: bool = False
    max_batch_size: int = 32 #Started with smaller to 4M batch size after specific tokens
    max_seq_len: int = 2048
    flash: bool = False # use flash attention?
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self,k, v)
        if self.n_kv_heads == None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads, "Numeber of Key/Value heads CANNOT be more than Number of Heads"
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0, "Make Sure to use NICE numbers to be distributed among Heads!"

#===============================ROOT MEAN SQR LAYER NORM================================
class RMSNORM(nn.Module):
    pass
#====================================BLOCK==============================================
class BLOCK(nn.Module):
    pass
#====================================LLAMA WRAPPER======================================
class LLama(nn.Module):
    def __init__(self, config, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model = nn.ModuleDict(dict(
            embed_tokens = nn.Embedding(config.vocab_size, config.dim),
            layers = nn.ModuleList(list(BLOCK(config) for _ in range(config.n_layers))),
            norm = RMSNORM(config.dim)
            ))
        self.lm_head = nn.Linear(config.dim, config.vocab_size)
    
    def forward():
        pass
    
    def generate():
        pass
    
    def text_completion():
        pass
    
