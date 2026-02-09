from autodiff.graph.data.receiver import Receiver
from autodiff.graph.tensor import Tensor
from autodiff.helper import benchmark
from autodiff.nn.normalization.layernorm import LayerNorm
import autodiff
import torch

def test_layernorm ():
    # Tests sum
    torch.manual_seed(42)

    x = torch.tensor([[5.0, 6.0, 3.0], [4.0, 5.0, 6.0]])  # Shape: (2, 3)
    x = Tensor.from_torch(x, requires_grad=True)

    ln = LayerNorm(3)
    result = ln(x)
    result.backward()

    def asrt (x, bias_g, scale_g):
        assert round(x.flatten()[0].item(),4) == 0.2673
        assert round(bias_g.flatten()[0].item(),4) == 2.0
        assert round(scale_g.flatten()[0].item(),4) == -0.9575

    Receiver(asrt, [
        result, 
        ln.bias.grad().contigious(), 
        ln.scale.grad().contigious(),
    ]);

    benchmark(lambda: autodiff.execute(), name="full exec")

if __name__ == "__main__":
    test_layernorm()

"""
Equivalent test in Pytorch:

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


"""
