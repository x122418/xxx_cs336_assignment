import torch


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)
    return

def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['iteration']
