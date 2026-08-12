from typing import Any

import torch
import torch.nn as nn
import math
from einops import einsum, rearrange


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # 旋转角度表 表示每一个维度对都有一个不同的基准旋转角度 （这里只是为了有更多的频率捕捉能力
        # 同时 不同位置的token旋转的倍数不同 是rope的核心 完成了相对位置编码的构建
        theta_ik_table = rearrange(torch.arange(max_seq_len, device=device), "max_S -> max_S 1") / (
            rearrange(
                theta ** ((2 * torch.arange(1, d_k // 2 + 1, device = device) - 2) / d_k),
                "half_max_s -> 1 half_max_s"
            )
        )
        # 表示所有的sin值 包括max_seq_len行 d_k//2 列  由于可以反复利用并且不会改动 所以注册为缓存 而非 参数
        self.register_buffer("sin_table", torch.sin(theta_ik_table), persistent=False)
        self.register_buffer("cos_table", torch.cos(theta_ik_table), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 在dim维度上 显式拆分为成对的两组
        x_tmp = rearrange(x, "... (half_d two) -> ... half_d two", two = 2)
        x_group1 = x_tmp[..., 0]
        x_group2 = x_tmp[..., 1]
        # 逐元素 完成旋转计算
        selected_sin_table = self.sin_table[token_positions]
        selected_cos_table = self.cos_table[token_positions]
        x_rotated_group1 = x_group1 * selected_cos_table - x_group2 * selected_sin_table
        x_rotated_group2 = x_group1 * selected_sin_table + x_group2 * selected_cos_table
        x_rot = torch.stack([x_rotated_group1, x_rotated_group2], dim = -1)

        return rearrange(x_rot, "... half_d two -> ... (half_d two)")
