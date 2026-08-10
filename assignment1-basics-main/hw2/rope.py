from typing import Any

import torch
import torch.nn as nn
import math
from einops import einsum

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        cos_theta_table = []
        sin_theta_table = []

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        return