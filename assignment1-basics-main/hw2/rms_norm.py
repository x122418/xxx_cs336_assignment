from typing import Any

import torch
import torch.nn as nn
from einops import einsum, rearrange


class RMSnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.rms_weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32) # B S d
        rms = torch.sqrt(einsum(x, x, "B S d, B S d -> B S")/self.d_model + self.eps)
        x = x/rearrange(rms, "B S -> B S 1")
        
        result = einsum(x, self.rms_weight, "B S d, d -> B S d")
        # Return the result in the original dtype
        return result.to(in_dtype)
