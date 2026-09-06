import torch, gc
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import RotaryEmbedding, TransformerBlock

# def checkpointed_blocks(
#         block: nn.Module,
#         x,
#         num_layers: int,
#         group_size: int,
# ):
#     for start in range(0, num_layers, group_size):
#         end = min(num_layers, start + group_size)
#         num_of_blocks = end - start

#         def run_group(x, count=num_of_blocks):
#             for _ in range(count):
#                 x = block(x)
#             return x

#         x = checkpoint(run_group, x, use_reentrant=False)
#     return x


def checkpointed_blocks(
    blocks,
    x,
    group_size: int,
):
    num_layers = len(blocks)

    for start in range(0, num_layers, group_size):
        end = min(start + group_size, num_layers)

        def run_group(x, start=start, end=end):
            for layer_index in range(start, end):
                x = blocks[layer_index](x)
            return x

        x = checkpoint(
            run_group,
            x,
            use_reentrant=False,
        )

    return x


# def mearuse_peak_memory(block: nn.Module, device, group_size:int):
#     batch_size = 4
#     context_length = 2048
#     d_model = 2560
#     num_layers = 32

#     block.zero_grad(set_to_none = True)
#     gc.collect()
#     torch.cuda.empty_cache()

#     x = torch.rand(
#         batch_size,
#         context_length,
#         d_model,
#         device=device,
#         requires_grad=True
#     )

#     torch.cuda.reset_peak_memory_stats(device)

#     y = checkpointed_blocks(
#         block,
#         x,
#         num_layers,
#         group_size
#     )

#     dummy_loss = y.sum()
#     dummy_loss.backward()
#     torch.cuda.synchronize(device)

#     peak_gib = (
#         torch.cuda.max_memory_allocated(device) / 1024**3
#     )

#     del dummy_loss
#     del y
#     del x

#     return peak_gib


def measure_peak_memory(
    blocks,
    device,
    group_size: int,
):
    batch_size = 4
    context_length = 2048
    d_model = 2560

    blocks.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    x = torch.randn(
        batch_size,
        context_length,
        d_model,
        device=device,
        requires_grad=True,
    )

    torch.cuda.reset_peak_memory_stats(device)

    y = checkpointed_blocks(
        blocks=blocks,
        x=x,
        group_size=group_size,
    )

    dummy_loss = y.sum()
    dummy_loss.backward()

    torch.cuda.synchronize(device)

    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3

    del dummy_loss
    del y
    del x

    blocks.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    return peak_gib


def main():
    device = "cuda:4"

    batch_size = 4
    context_length = 2048
    d_model = 2560
    d_ff = 10240
    num_heads = 16
    num_layers = 32


    # block = TransformerBlock(
    #     d_model=d_model, d_ff=d_ff, num_heads=num_heads, positional_encoder=rotary
    # ).to(device)

    # compiled_block = torch.compile(block, fullgraph=True)

    blocks = nn.ModuleList(
        [torch.compile(
            TransformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                num_heads=num_heads,
                positional_encoder=RotaryEmbedding(
                    context_length=context_length,
                    dim=d_model // num_heads,
                ),
            ).to(device),
            fullgraph=True,)
        for _ in range(num_layers)
        ]
    )

    print("Warming up 32 compiled blocks...")

    for layer_index, layer in enumerate(blocks):
        warmup_x = torch.randn(
            batch_size,
            context_length,
            d_model,
            device=device,
            requires_grad=True,
        )

        warmup_y = layer(warmup_x)
        warmup_y.sum().backward()

        layer.zero_grad(set_to_none=True)

        del warmup_y
        del warmup_x

        print(f"Warmed up layer {layer_index}")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    for group_size in [1, 2, 4]:
        peak_gib = measure_peak_memory(
            blocks=blocks,
            device=device,
            group_size=group_size,
        )

        print(f"group_size={group_size}, " f"peak_memory={peak_gib:.2f} GiB")


if __name__ == "__main__":
    main()
