from jaxtyping import Bool, Float, Int
import torch
from einops import reduce
def cross_entropy(
    inputs: Float[torch.Tensor, " batch_size vocab_size"], targets: Int[torch.Tensor, " batch_size"]
) -> Float[torch.Tensor, ""]:
    # 这里的batch_size 实际上是 B*S 把每个token看作单独样本
    modified_inputs = inputs - torch.amax(inputs, dim = -1, keepdim=True)
    selected = torch.gather(modified_inputs, dim = -1, index=targets.unsqueeze(-1)).squeeze(-1)
    log_softmax = selected - torch.log(torch.sum(modified_inputs.exp(), dim = -1))

    res = reduce(-log_softmax, "batch_size -> ","mean")
    return res