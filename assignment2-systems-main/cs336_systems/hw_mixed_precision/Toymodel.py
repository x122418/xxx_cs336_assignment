import torch.nn as nn
import torch

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        print("fc1_out:", x.dtype)

        x = self.relu(x)
        print("relu_out:", x.dtype)

        x = self.ln(x)
        print("ln_out:", x.dtype)

        x = self.fc2(x)

        print("fc2_out:", x.dtype)
       
        return x

if __name__ == "__main__":
    device = "cuda:7"
    toy = ToyModel(in_features=16, out_features=5).to(device)
    
    x = torch.rand(4, 16, device = device)
    targets = torch.randint(0, 5, (4,), device = device)

    with torch.autocast(device_type = "cuda", dtype = torch.float16):
        print("parameter:", toy.fc1.weight.dtype)

        logits = toy(x)
        print("logits:", logits.dtype)

        loss = torch.nn.functional.cross_entropy(logits, targets)
        print("loss:", loss.dtype)

    loss.backward()
    for name, parameter in toy.named_parameters():
        print(name, "gradient:", parameter.grad.dtype)