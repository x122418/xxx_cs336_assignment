from typing import Any

import torch
from torch import nn
from einops import einsum, rearrange
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import (
    RotaryEmbedding,
    TransformerBlock,
)


class RMSnorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5, device=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.device = device
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # B S d
        rms = torch.rsqrt(
            einsum(x, x, "B S d, B S d -> B S") / self.hidden_size + self.eps
        )
        x = x * rearrange(rms, "B S -> B S 1")

        result = einsum(x, self.weight, "B S d, d -> B S d")
        return result.to(in_dtype)


def pack_hook(tensor):
    global total_size_bytes
    global saved_tensor_records

    if not isinstance(tensor, nn.Parameter):
        tensor_size = tensor.numel() * tensor.element_size()
        total_size_bytes += tensor_size

        saved_tensor_records.append(
            {
                "shape": tensor.shape,
                "dtype": tensor.dtype,
                "size_bytes": tensor_size,
                "grad_fn": tensor.grad_fn
            }
        )

    print(
        "Saving_residuals:",
        f"shape: {tensor.shape}",
        f"dtype: {tensor.dtype}",
        f"grad_fn: {tensor.grad_fn}",
    )
    return tensor


def unpack_hook(tensor):
    print(
        "Loading_residuals:",
        f"shape: {tensor.shape}",
        f"dtype: {tensor.dtype}",
        f"grad_fn: {tensor.grad_fn}",
    )
    return tensor


def main():
    device = "cuda:4"
    global total_size_bytes
    global saved_tensor_records

    def four_blocks(x):
        x = compiled_block(x)
        x = compiled_block(x)
        x = compiled_block(x)
        x = compiled_block(x)
        return x

    def two_blocks(x):
        x = compiled_block(x)
        x = compiled_block(x)
        return x

    def four_blocks_checkpoint(x):
        x = checkpoint(two_blocks, 
                       x, 
                       use_reentrant=False)
        x = checkpoint(two_blocks, 
                               x, 
                               use_reentrant=False)
        return x


    batch_size = 4
    context_length = 2048
    d_model = 2560
    d_ff = 10240
    num_heads = 16

    rotary = RotaryEmbedding(
        context_length=context_length,
        dim=d_model // num_heads,
    )

    block = TransformerBlock(
        d_model=d_model, d_ff=d_ff, num_heads=num_heads, positional_encoder=rotary
    ).to(device)

    compiled_block = torch.compile(block, fullgraph=True)

    warmup_x = torch.randn(
        batch_size,
        context_length,
        d_model,
        device=device,
        requires_grad=True,
    )
    compiled_block(warmup_x).sum().backward()
    block.zero_grad(set_to_none=True)
    del warmup_x

    # 正式记录：只执行 forward，观察为未来 backward 保存的 residuals
    x = torch.randn(
        batch_size,
        context_length,
        d_model,
        device=device,
        requires_grad=True,
    )

    total_size_bytes = 0
    saved_tensor_records = []

    with torch.autograd.graph.saved_tensors_hooks(
        pack_hook,
        unpack_hook,
    ):
        # y = compiled_block(x)
        # y = four_blocks(x)
        y = four_blocks_checkpoint(x)

    largest = sorted(
        saved_tensor_records,
        key = lambda item: item['size_bytes'],
        reverse=True
    )[:5]

    print("\nFive largest saved tensor")

    for idx, item in enumerate(largest, start = 1):
        percentage = (
            item['size_bytes'] / total_size_bytes * 100
        )
        print(
        f"{idx}. "
        f"shape={item['shape']}, "
        f"size={item['size_bytes'] / 1024**2:.2f} MiB, "
        f"percentage={percentage:.2f}%"
    )

    print(
        f"Total logical size: "
        f"{total_size_bytes / 1024**2: .2f} MiB"
    )


#     rms_norm 对应实验
#     rms_norm = RMSnorm(hidden_size=2560, device=device)
#     compiled_rms_norm = torch.compile(rms_norm)
#     # 第一次：触发编译，不记录 saved tensors
#     warmup_x = torch.randn(
#         4,
#         512,
#         2560,
#         device=device,
#         requires_grad=True,
#     )
#     warmup_y = compiled_rms_norm(warmup_x)
#     warmup_y.sum().backward()

#     x = torch.rand(4, 512, 2560,
#                 device = device,
#                 requires_grad=True)

#     with torch.autograd.graph.saved_tensors_hooks(
#         pack_hook,
#         unpack_hook,
#     ):
#         y = compiled_rms_norm(x)
#         loss = y.sum()
#         loss.backward()
#     print(
#     "Total logical size:",
#     f"{total_size_bytes / 1024**2:.2f} MiB",
# )


if __name__ == "__main__":
    main()
