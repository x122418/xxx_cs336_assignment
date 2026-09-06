from cs336_basics.model import (
    RotaryEmbedding,
    TransformerBlock,
)

import torch, math
import torch.nn as nn
import einops
from einops import rearrange, einsum

D_VALUES = [16, 32, 64, 128]
S_VALUES = [256, 512, 1024, 2048]
batch_size = 4
warmup_steps = 5
measurement_steps = 10
device = "cuda:7"



def pytorch_attention(q, k, v):
    attn_scores = einsum(q, k, "b s d, b t d -> b s t")
    attn_scores = attn_scores / math.sqrt(q.shape[-1])
    attn_scores = torch.softmax(attn_scores, dim = -1)
    output = attn_scores @ v
    return output

def measure_cuda_time(fn, measure_steps):
    times_ms = []
    for _ in range(measure_steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        result = fn()
        end.record()

        end.synchronize()
        times_ms.append(start.elapsed_time(end))
        del result

    return sum(times_ms)/len(times_ms)


def main():
    for context_length in S_VALUES:
        for d in  D_VALUES:

        q = torch.randn(
            batch_size, 
            context_length,
            d,
            device = device,
            dtype = torch.float32,
            requires_grad = True,
        )

        k = torch.randn(
            batch_size, 
            context_length,
            d,
            device = device,
            dtype = torch.float32,
            requires_grad = True,
        )

        v = torch.randn(
            batch_size, 
            context_length,
            d,
            device = device,
            dtype = torch.float32,
            requires_grad = True,
        )

        # warm up
        for _ in range(warmup_steps):
            output = pytorch_attention(q, k, v)
        torch.cuda.synchronize()

        forward_fn = lambda: pytorch_attention(q, k, v)

        # fromal measurement
        time_avg = measure_cuda_time(forward_fn, measure_steps)



if __name__ == "__main__" : 
    main()