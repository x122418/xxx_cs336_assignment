import numpy as np
import numpy.typing as npt
import torch


def data_loading(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
):
    seq_len = len(dataset)
    # 可能的起点情况数 (需要同时考虑到inputs和targets合法)
    start_pos = seq_len - context_length
    starts = np.random.randint(0, start_pos, batch_size)
    offsets = np.arange(0, context_length)
    indices = starts[:,None] + offsets[None,:]
    inputs = torch.tensor(dataset[indices], dtype = torch.long, device = device)
    targets = torch.tensor(dataset[indices+1], dtype = torch.long, device = device)

    return (inputs, targets)
