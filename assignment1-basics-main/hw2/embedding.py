import torch
import torch.nn as nn
import math
from einops import einsum


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        embedding_mapping = torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype
        )
        self.weight = nn.Parameter(embedding_mapping)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        return self.weight[token_ids]