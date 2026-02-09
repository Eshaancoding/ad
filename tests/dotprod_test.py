from autodiff.graph.data.receiver import Receiver
from autodiff.graph.tensor import Tensor
from autodiff.helper import benchmark
import autodiff
import torch

def assert_results (result, input_grad):
    assert round(result.sum().item(), 4) == -3.2747
    assert round(input_grad.sum().item(), 4) == 8.2885

def test_dotprod ():
    torch.manual_seed(42)
    inp = Tensor.from_torch(torch.randn(2,5), requires_grad=True)
    weight = Tensor.from_torch(torch.randn(5,3), requires_grad=True)
    bias = Tensor.from_torch(torch.randn(3), requires_grad=True)

    result = (inp @ weight) + bias
    result.backward()

    Receiver(assert_results, [result, inp.grad()], name="saving params")
    benchmark(lambda: autodiff.execute(), name="full exec")

if __name__ == "__main__":
    test_dotprod()


"""
Equivalent test in Pytorch:

import torch

def assert_results(result, input_grad):
    print(result.sum().item())
    print(input_grad.sum().item())
    # Uncomment asserts to verify expected values if desired
    # assert round(result.sum().item(), 5) == 6.2689
    # assert round(input_grad.sum().item(), 5) == 5.39896

def test_dotprod():
    torch.manual_seed(42)  # Set seed for reproducibility
    
    inp = torch.randn(2, 5, requires_grad=True)
    weight = torch.randn(5, 3, requires_grad=True)
    bias = torch.randn(3, requires_grad=True)
    result = inp @ weight + bias  

    result.sum().backward()

    print(inp, weight, bias)

    assert_results(result, inp.grad)

if __name__ == "__main__":
    test_dotprod()


"""
