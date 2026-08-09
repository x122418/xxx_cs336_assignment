import torch
import torch.nn as nn
import math
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # init weight matrix
        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.weight = nn.Parameter(W)
        sigma = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return einsum(
            x,
            self.weight,
            "... in_feature, out_feature in_feature -> ... out_feature",
        )
