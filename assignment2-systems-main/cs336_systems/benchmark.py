from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.nn_utils import softmax

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
    measurement_steps = benchmark_cfg["measurement_steps"]

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

    inputs = torch.randint(
        0, vocab_size, (batch_size, sequence_length), dtype=torch.long, device=device
    )
    targets = torch.randint(
        0, vocab_size, (batch_size, sequence_length), dtype=torch.long, device=device
    )

    torch.cuda.synchronize()
    if mode == "forward":
        with torch.inference_mode():
            for _ in range(warmup_steps):
                outputs = lm_model(inputs)
                torch.cuda.synchronize()
                del outputs
            elapsed_times = []

            with nvtx.range("profile_region"):
                for _ in range(measurement_steps):
                    torch.cuda.synchronize()
                    t1 = timeit.default_timer()
                    with nvtx.range('forward'):
                        outputs = lm_model(inputs)
                    torch.cuda.synchronize()
                    t2 = timeit.default_timer()
                    gap_time = (t2 - t1) * 1000
                    del outputs
                    elapsed_times.append(gap_time)

            time_mean = np.mean(elapsed_times)
            time_std = np.std(elapsed_times)

            print(time_mean)
            print(time_std)

    elif mode == "forward_backward":
        for _ in range(warmup_steps):
            lm_model.zero_grad(set_to_none=True)
            outputs = lm_model(inputs)
            loss = cross_entropy(outputs, targets)
            loss.backward()

            torch.cuda.synchronize()
            del outputs, loss
        elapsed_times = []
        with nvtx.range("profile_region"):
            for _ in range(measurement_steps):
                torch.cuda.synchronize()
                t1 = timeit.default_timer()
                lm_model.zero_grad(set_to_none=True)

                with nvtx.range("forward"):
                    outputs = lm_model(inputs)
                with nvtx.range("loss"):
                    loss = cross_entropy(outputs, targets)
                with nvtx.range("backward"):
                    loss.backward()

                torch.cuda.synchronize()
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
            outputs = lm_model(inputs)
            loss = cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            del outputs, loss
        elapsed_times = []
        with nvtx.range("profile_region"):
            for _ in range(measurement_steps):
                torch.cuda.synchronize()
                t1 = timeit.default_timer()
                optimizer.zero_grad(set_to_none=True)
                with nvtx.range("forward"):
                    outputs = lm_model(inputs)
                with nvtx.range("loss"):
                    loss = cross_entropy(outputs, targets)
                with nvtx.range("backward"):
                    loss.backward()
                with nvtx.range("update"):
                    optimizer.step()

                torch.cuda.synchronize()
                t2 = timeit.default_timer()
                gap_time = (t2 - t1) * 1000
                del outputs, loss
                elapsed_times.append(gap_time)

        time_mean = np.mean(elapsed_times)
        time_std = np.std(elapsed_times)

        print(time_mean)
        print(time_std)
    else:
        raise ValueError(f"Unsupported benchmark mode: {mode}")

    res = {
        'model_size': model_size,
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
