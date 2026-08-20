from typing import Any

from .softmax import softmax
import math
from einops import einsum, rearrange
import torch
import torch.nn as nn
from .linear import Linear
from .rope import RotaryPositionalEmbedding
from .SwiGLU import SwiGLU
from .rms_norm import RMSnorm
from .embedding import Embedding
from .softmax import softmax


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
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)

    def forward(
        self, x: torch.Tensor, token_positions=None
    ):  # token_positions (batch_size, sequence_length)
        seq_len = x.shape[-2]
        # 构造因果掩码
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        # 对于kqv分别拆分多头
        q = rearrange(
            self.q_proj(x), "... s (num d) -> ... num s d", num=self.num_heads
        )
        k = rearrange(
            self.k_proj(x), "... s (num d) -> ... num s d", num=self.num_heads
        )
        v = rearrange(
            self.v_proj(x), "... s (num d) -> ... num s d", num=self.num_heads
        )
        # 位置编码
        if self.use_rope:
            q = self.position_embed(q, token_positions)
            k = self.position_embed(k, token_positions)
        # self_attention   (... num d)
        attention = scaled_dot_product_attention(q, k, v, causal_mask)
        attention_o = rearrange(
            attention, "... num s d -> ... s (num d)", num=self.num_heads
        )
        output = self.output_proj(attention_o)
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        use_rmsnorm: bool = True
    ) -> None:
        super().__init__()
        self.attn = multihead_self_attention(d_model, num_heads, theta, max_seq_len)
        self.ffn = SwiGLU(d_model, d_ff)
        if use_rmsnorm:
            self.ln1 = RMSnorm(d_model)
            self.ln2 = RMSnorm(d_model)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,  # (batch_size, seq_len, d_model)
    ):
        x = x + self.attn(self.ln1(x))
        output = x + self.ffn(self.ln2(x))

        return output


class Transformer_LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        rope_theta: float,
        use_rmsnorm: bool = True

    ) -> None:
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, use_rmsnorm=use_rmsnorm)
            for _ in range(num_layers)
        )
        if use_rmsnorm:
            self.ln_final = RMSnorm(d_model)
        else:
            self.ln_final = nn.Identity()
        self.lm_head = Linear(d_model, vocab_size)


    def forward(
        self,
        token_ids: torch.Tensor # B S
    ):
        embeddings = self.token_embeddings(token_ids) # B S D
        embeddings_attn = embeddings
        for layer in self.layers:
            embeddings_attn = layer(embeddings_attn)     # B S D
        output = self.lm_head(self.ln_final(embeddings_attn))  # B S vocab_size

        return output
