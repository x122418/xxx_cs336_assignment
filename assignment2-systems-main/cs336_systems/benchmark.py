from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy, clip_gradient
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from contextlib import nullcontext
from torch import Tensor
from cs336_basics.nn_utils import softmax
from pathlib import Path

import argparse
import numpy as np
import math
import torch
import timeit
import yaml
import random
import json
import torch.cuda.nvtx as nvtx
import cs336_basics.model as basic_model

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--model_size", type=str, required=True)
parser.add_argument("--mode", type=str, required=True)
parser.add_argument("--warmup_steps", type=int, required=True)
parser.add_argument("--sequence_length", type=int, default=None)
parser.add_argument("--annotate_attention", action="store_true")
parser.add_argument("--profile_memory", action="store_true")
parser.add_argument("--memory_snapshot_path", type = str, default=None)
parser.add_argument(
    "--pytorch_nvtx",
    action="store_true",
)

args = parser.parse_args()


def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    with nvtx.range("attention"):
        d_k = K.shape[-1]
        with nvtx.range("attention_qk_matmul"):
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") 
        with nvtx.range("attention_scale"):
            attention_scores = attention_scores / math.sqrt(d_k)

        if mask is not None:
            with nvtx.range("attention_mask"):
                attention_scores = torch.where(mask, attention_scores, float("-inf"))
        with nvtx.range("attention_softmax"):
            attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension
        with nvtx.range("attention_av_matmul"):
            attention_output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return attention_output

def add_transformer_block_nvtx_ranges(model):
    def make_wrapped_forward(forward_fn, index):
        def wrapped_forward(*inputs, **kwargs):
            with nvtx.range(f"TransformerBlock_{index}"):
                return forward_fn(*inputs, **kwargs)
        return wrapped_forward
    
    for block_idx, block in enumerate(model.layers):
        original_forward = block.forward

        block.forward = make_wrapped_forward(
            original_forward, 
            block_idx,
        )

def main():
    # 读取超参数配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_size = args.model_size
    mode = args.mode
    warmup_steps = args.warmup_steps

    experiment_cfg = config["experiment"]
    model_cfg = config["models"][model_size]
    benchmark_cfg = config["benchmark"]
    runtime_cfg = config["runtime"]
    shared_model_cfg = config['shared_model']

    # 设置random seed和device
    seed = experiment_cfg["seed"]
    max_grad_norm = benchmark_cfg.get("max_grad_norm", None)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(runtime_cfg["device"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Using device: {device}")

    # 构建语言model实例
    vocab_size = shared_model_cfg["vocab_size"]
    d_model = model_cfg["d_model"]
    num_heads = model_cfg["num_heads"]
    max_seq_len = shared_model_cfg["max_seq_len"]

    batch_size = benchmark_cfg["batch_size"]
    sequence_length = (
    args.sequence_length
    if args.sequence_length is not None
    else benchmark_cfg["sequence_length"]
)
    if args.profile_memory and args.memory_snapshot_path is None:
        raise ValueError(
            "--profile_memory requires --memory_snapshot_path"
        )

    if args.profile_memory:
        snapshot_path = Path(args.memory_snapshot_path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    
    measurement_steps = benchmark_cfg["measurement_steps"]
    precision = benchmark_cfg.get("precision", "fp32")
    if precision not in ("fp32", "bf16"):
        raise ValueError(f"Unsupported precision: {precision}")
    def precision_context():
        if precision == "bf16":
            return torch.autocast(
                device_type = "cuda",
                dtype = torch.bfloat16,
            )
        return nullcontext()
    
    def pytorch_nvtx_context():
        if args.pytorch_nvtx:
            return torch.autograd.profiler.emit_nvtx(
                record_shapes=True
            )
        return nullcontext()

    if args.annotate_attention:
        basic_model.scaled_dot_product_attention =(
            annotated_scaled_dot_product_attention
        )

    lm_model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=max_seq_len,
        d_model=d_model,
        num_layers=model_cfg["num_layers"],
        num_heads=num_heads,
        d_ff=model_cfg["d_ff"],
        rope_theta=shared_model_cfg["theta"],
    ).to(device)

    if args.pytorch_nvtx:
        add_transformer_block_nvtx_ranges(lm_model)

    inputs = torch.randint(
        0, vocab_size, (batch_size, sequence_length), dtype=torch.long, device=device
    )
    targets = torch.randint(
        0, vocab_size, (batch_size, sequence_length), dtype=torch.long, device=device
    )

    torch.cuda.synchronize(device)
    if mode == "forward":
        with torch.inference_mode():
            for _ in range(warmup_steps):
                with precision_context():
                    outputs = lm_model(inputs)
                torch.cuda.synchronize(device)
                del outputs
            elapsed_times = []

            if args.profile_memory:
                torch.cuda.memory._record_memory_history(max_entries = 1_000_000)

            with nvtx.range("profile_region"):
                for _ in range(measurement_steps):
                    torch.cuda.synchronize(device)

                    
                    with nvtx.range('forward'):
                        
                        t1 = timeit.default_timer()
                        with precision_context():
                            outputs = lm_model(inputs)
                        torch.cuda.synchronize(device)
                        t2 = timeit.default_timer()
                    
                    
                    gap_time = (t2 - t1) * 1000
                    del outputs
                    elapsed_times.append(gap_time)

            if args.profile_memory:
                torch.cuda.synchronize(device)
                torch.cuda.memory._dump_snapshot(str(snapshot_path))
                torch.cuda.memory._record_memory_history(enabled=None)

            time_mean = np.mean(elapsed_times)
            time_std = np.std(elapsed_times)

            print(time_mean)
            print(time_std)

    elif mode == "forward_backward":
        for _ in range(warmup_steps):
            lm_model.zero_grad(set_to_none=True)
            with precision_context():
                outputs = lm_model(inputs)
                loss = cross_entropy(outputs, targets)
            loss.backward()
            print("loss:", loss.item())
            print(
                "grad finite:", 
                all(
                    torch.isfinite(p.grad).all().item()
                    for p in lm_model.parameters()
                    if p.grad is not None
                )
            )

            torch.cuda.synchronize(device)
            del outputs, loss
        elapsed_times = []
        with nvtx.range("profile_region"):
            for _ in range(measurement_steps):
                torch.cuda.synchronize(device)
                t1 = timeit.default_timer()
                lm_model.zero_grad(set_to_none=True)

                with precision_context():
                    with nvtx.range("forward"):
                        outputs = lm_model(inputs)
                    with nvtx.range("loss"):
                        loss = cross_entropy(outputs, targets)
                with nvtx.range("backward"):
                    loss.backward()

                torch.cuda.synchronize(device)
                t2 = timeit.default_timer()
                gap_time = (t2 - t1) * 1000
                del outputs, loss
                elapsed_times.append(gap_time)

        time_mean = np.mean(elapsed_times)
        time_std = np.std(elapsed_times)

        print(time_mean)
        print(time_std)

    elif mode == "train_step":
        optimizer = AdamW(lm_model.parameters(), lr=1e-3)
        for _ in range(warmup_steps):
            optimizer.zero_grad(set_to_none=True)
            with precision_context():
                outputs = lm_model(inputs)
                loss = cross_entropy(outputs, targets)
            loss.backward()
            if max_grad_norm is not None:
                with nvtx.range("clip_gradient"):
                    clip_gradient(lm_model.parameters(), max_grad_norm)
            optimizer.step()

            torch.cuda.synchronize(device)
            del outputs, loss
        elapsed_times = []

        if args.profile_memory:
            torch.cuda.memory._record_memory_history(max_entries = 1_000_000)

        with pytorch_nvtx_context():
            with nvtx.range("profile_region"):
                for _ in range(measurement_steps):
                    torch.cuda.synchronize(device)
                    t1 = timeit.default_timer()
                    optimizer.zero_grad(set_to_none=True)
                    with precision_context():
                        with nvtx.range("forward"):
                            outputs = lm_model(inputs)
                        with nvtx.range("loss"):
                            loss = cross_entropy(outputs, targets)
                    
                    with nvtx.range("backward"):
                        loss.backward()

                    if max_grad_norm is not None:
                        with nvtx.range("clip_gradient"):
                            clip_gradient(lm_model.parameters(), max_grad_norm)

                    with nvtx.range("update"):
                        optimizer.step()

                    torch.cuda.synchronize(device)
                    t2 = timeit.default_timer()
                    gap_time = (t2 - t1) * 1000
                    del outputs, loss
                    elapsed_times.append(gap_time)

        if args.profile_memory:
            torch.cuda.synchronize(device)
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            torch.cuda.memory._record_memory_history(enabled=None)

        time_mean = np.mean(elapsed_times)
        time_std = np.std(elapsed_times)

        print(time_mean)
        print(time_std)
    else:
        raise ValueError(f"Unsupported benchmark mode: {mode}")

    res = {
        'model_size': model_size,
        "precision": precision,
        'mode': mode,
        'd_model': d_model,
        'd_ff': model_cfg["d_ff"],
        'num_layers': model_cfg["num_layers"],
        'num_heads':num_heads,
        'batch_size':batch_size,
        'sequence_length':sequence_length,
        'warmup_steps':warmup_steps,
        'measurement_steps':measurement_steps,
        'mean_ms':float(time_mean),
        'std_ms':float(time_std),
        'status':"ok",
    }
    return res

if __name__ == "__main__":
    result = main()
    print(json.dumps(result))

