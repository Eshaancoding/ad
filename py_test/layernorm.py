import torch
import torch.nn as nn
import time

def test_layernorm():
    torch.manual_seed(42)
    
    # Input
    x = torch.tensor([[5.0, 6.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)  # Shape: (2, 3)
    layernorm = nn.LayerNorm(3, eps=1e-5)  # has learnable bias and scale
    result = layernorm(x)
    result.sum().backward()

    print(result)
    print(layernorm.bias.grad)
    print(layernorm.weight.grad)
    
if __name__ == "__main__":
    test_layernorm()
