import torch, math
import torch.nn as nn
import einops
import gc
from einops import rearrange, einsum

D_VALUES = [16, 32, 64, 128]
S_VALUES = [256, 1024, 4096, 8192, 16384]
measurement_steps = 100

batch_size = 8
warmup_steps = 5
device = "cuda:1"


def pytorch_attention(q, k, v):
    attn_scores = einsum(q, k, "b s d, b t d -> b s t")
    attn_scores = attn_scores / math.sqrt(q.shape[-1])
    probabilities = torch.softmax(attn_scores, dim = -1)
    output = probabilities @ v
    return output

compiled_attention = torch.compile(
    pytorch_attention,
    fullgraph=True,
)

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

def measure_backward_time(
    attention_fn,
    q,
    k,
    v,
    measurement_steps,
):
    times_ms = []
    output_grad = torch.randn_like(q)

    for _ in range(measurement_steps):
        # Forward 不计入 backward 时间，但用于建立计算图
        output = attention_fn(q, k, v)
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output.backward(output_grad)
        end.record()
        end.synchronize()

        times_ms.append(start.elapsed_time(end))

        q.grad = None
        k.grad = None
        v.grad = None
        del output

    del output_grad
    return sum(times_ms) / len(times_ms)


def measure_forward_memory(q, k, v, device):
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    baseline_bytes = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    output = pytorch_attention(q, k, v)
    torch.cuda.synchronize(device)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    extra_bytes = peak_bytes - baseline_bytes

    del output
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return (
        peak_bytes / 1024**3,
        extra_bytes / 1024**3
    )

def measure_forward_backward_memory(q, k, v, device):
    q.grad = None
    k.grad = None
    v.grad = None

    output_grad = torch.randn_like(q)

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    baseline_bytes = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    output = pytorch_attention(q, k, v)
    output.backward(output_grad)

    torch.cuda.synchronize(device)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    extra_bytes = peak_bytes - baseline_bytes

    del output
    del output_grad

    q.grad = None
    k.grad = None
    v.grad = None

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    return (
        peak_bytes / 1024**3,
        extra_bytes / 1024**3,
    )

def run_one_config(context_length, d):
    q = torch.randn(
        batch_size,
        context_length,
        d,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    k = torch.randn(
        batch_size,
        context_length,
        d,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    v = torch.randn(
        batch_size,
        context_length,
        d,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    # Forward warmup
    for _ in range(warmup_steps):
        output = pytorch_attention(q, k, v)

    torch.cuda.synchronize(device)
    del output


    # Formal forward measurement
    eager_forward_ms = measure_cuda_time(
        lambda: pytorch_attention(q, k, v),
        measurement_steps,
    )

    # Backward warmup
    output_grad = torch.randn_like(q)

    for _ in range(warmup_steps):
        output = pytorch_attention(q, k, v)
        output.backward(output_grad)

        q.grad = None
        k.grad = None
        v.grad = None

        del output

    torch.cuda.synchronize(device)
    del output_grad

    q.grad = None
    k.grad = None
    v.grad = None

    # Formal backward measurement
    eager_backward_ms = measure_backward_time(
        pytorch_attention,
        q,
        k,
        v,
        measurement_steps,
    )

    # Forward peak memory
    forward_peak_gib, forward_extra_gib = (
        measure_forward_memory(
            q,
            k,
            v,
            device,
        )
    )

    # Forward + backward peak memory
    (
        forward_backward_peak_gib,
        forward_backward_extra_gib,
    ) = measure_forward_backward_memory(
        q,
        k,
        v,
        device,
    )


    # 2. torch.compile
    # Compiled forward warmup
    for _ in range(warmup_steps):
        output = compiled_attention(q, k, v)

    torch.cuda.synchronize(device)
    del output

    compiled_forward_ms = measure_cuda_time(
        lambda: compiled_attention(q, k, v),
        measurement_steps,
    )

    # Compiled backward warmup
    output_grad = torch.randn_like(q)

    for _ in range(warmup_steps):
        output = compiled_attention(q, k, v)
        output.backward(output_grad)

        q.grad = None
        k.grad = None
        v.grad = None
        del output

    torch.cuda.synchronize(device)
    del output_grad

    compiled_backward_ms = measure_backward_time(
        compiled_attention,
        q,
        k,
        v,
        measurement_steps,
    )

    result = {
        "sequence_length": context_length,
        "d": d,
        "eager_forward_ms": eager_forward_ms,
        "eager_backward_ms": eager_backward_ms,
        "compiled_forward_ms": compiled_forward_ms,
        "compiled_backward_ms": compiled_backward_ms,
        "forward_peak_gib": forward_peak_gib,
        "forward_extra_gib": forward_extra_gib,
        "forward_backward_peak_gib": forward_backward_peak_gib,
        "forward_backward_extra_gib": forward_backward_extra_gib,
        "status": "ok",
    }


    del q
    del k
    del v

    return result


def main():
    torch.cuda.set_device(device)

    for context_length in S_VALUES:
        for d in D_VALUES:
            print(
                f"Starting S={context_length}, d={d}",
                flush=True,
            )

            try:
                result = run_one_config(
                    context_length,
                    d,
                )

                print(
                    f"S={result['sequence_length']}, "
                    f"d={result['d']}, "
                    f"eager_forward={result['eager_forward_ms']:.4f} ms, "
                    f"compiled_forward={result['compiled_forward_ms']:.4f} ms, "
                    f"eager_backward={result['eager_backward_ms']:.4f} ms, "
                    f"compiled_backward={result['compiled_backward_ms']:.4f} ms, "
                    f"forward_extra={result['forward_extra_gib']:.4f} GiB, "
                    f"forward_backward_extra="
                    f"{result['forward_backward_extra_gib']:.4f} GiB, "
                    f"status={result['status']}",
                    flush=True,
                )

            except torch.OutOfMemoryError:
                print(
                    f"S={context_length}, "
                    f"d={d}, "
                    f"status=oom",
                    flush=True,
                )
            finally:
                torch.cuda.synchronize(device)
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__" : 
    main()