from jaxtyping import Bool, Float, Int
import torch
def softmax(in_features:Float[torch.Tensor, "..."], dim:int) -> torch.Tensor:
    # (1, 4, 9, 15)   exp(vec) / sum(exp(vec))
    down_input = in_features - torch.amax(in_features, dim = dim, keepdim=True)
    exp_down_input = down_input.exp()
    ans = exp_down_input / exp_down_input.sum(dim = dim, keepdim=True)

    return ans