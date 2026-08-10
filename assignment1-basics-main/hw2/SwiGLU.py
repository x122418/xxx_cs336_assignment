from typing import Any

import torch
import torch.nn as nn
from einops import einsum, rearrange
from .linear import Linear

def silu(x:torch.Tensor) -> torch.Tensor:
    return x / (1 + torch.exp(-x))

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff = None, device = None, dtype = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8/3 * d_model)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
