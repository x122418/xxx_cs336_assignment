from typing import Any

from .softmax import softmax
import math
from einops import einsum, rearrange
import torch
import torch.nn as nn
from .linear import Linear
from .rope import RotaryPositionalEmbedding


def scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        weights = scores.masked_fill(mask == 0, float("-inf"))
    else:
        weights = scores
    output = softmax(weights, dim=-1) @ V

    return output


class multihead_self_attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta=None,
        max_seq_len=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.device = device
        self.use_rope = False
        if (theta is not None) and (max_seq_len is not None):
            self.use_rope = True
            self.position_embed = RotaryPositionalEmbedding(
                theta, self.d_head, max_seq_len, device
            )
        self.w_q = Linear(d_model, d_model, device, dtype)
        self.w_k = Linear(d_model, d_model, device, dtype)
        self.w_v = Linear(d_model, d_model, device, dtype)
        self.w_o = Linear(d_model, d_model, device, dtype)

    def forward(self, x: torch.Tensor, token_positions=None):
        seq_len = x.shape[-2]
        # 构造因果掩码
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        # 对于kqv分别拆分多头
        q = rearrange(self.w_q(x), "... s (num d) -> ... num s d", num=self.num_heads)
        k = rearrange(self.w_k(x), "... s (num d) -> ... num s d", num=self.num_heads)
        v = rearrange(self.w_v(x), "... s (num d) -> ... num s d", num=self.num_heads)
        # 位置编码
        if self.use_rope:
            q = self.position_embed(q, token_positions)
            k = self.position_embed(k, token_positions)
        # self_attention   (... num d)
        attention = scaled_dot_product_attention(q, k, v, causal_mask)
        attention_o = rearrange(
            attention, "... num s d -> ... s (num d)", num=self.num_heads
        )
        output = self.w_o(attention_o)
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
    ) -> None:
        super().__init__()
        

    def forward(
        self,
        x: torch.Tensor, # (batch_size, seq_len, d_model)
    ):
        return
