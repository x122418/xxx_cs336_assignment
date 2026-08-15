from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas = (0.9, 0.999), eps=1e-9, weight_decay = 0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))

                alpha_t = lr * math.sqrt(1-beta2**t) / (1 - beta1**t)
                p.data -= lr * weight_decay * p.data
                m = beta1 * m + (1-beta1) * grad
                v = beta2 * v + (1-beta2) * grad**2
                t += 1
                p.data -= alpha_t * m / (eps + torch.sqrt(v))

                state["m"] = m
                state["t"] = t
                state["v"] = v
        return loss