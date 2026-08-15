from math import cos, pi
import torch
from collections.abc import Callable, Iterable
from einops import reduce
import math

def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        alpha_t = it / warmup_iters * max_learning_rate
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        alpha_t = min_learning_rate + 0.5 * (
            1 + cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * pi)
        ) * (max_learning_rate - min_learning_rate)
    else:
        alpha_t = min_learning_rate

    return alpha_t

def grad_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    parameters = list(parameters)
    
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return

    total_norm = math.sqrt(
        sum(torch.sum(g**2) for g in grads)
    )
    clip_factor = max_l2_norm/(total_norm + eps)
    if clip_factor < 1:
        for grad in grads:
            grad*=clip_factor

    return 