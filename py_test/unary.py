import torch
from time import time

def assert_results(result, result_two, input_grad):
    print("Forward one")
    print(result.sum())
    print("Forward two:")
    print(result_two.sum())
    print("Input gradient:")
    print(input_grad.sum())

def test_unary():
    # Input tensor with requires_grad=True
    inp = torch.tensor([
        [0.3, 0.2, 0.7, 0.54],
        [0.13, 0.77, 0.23, 0.22]
    ], dtype=torch.float32, requires_grad=True)

    result = torch.pow(2, inp) + 3
    result = torch.sqrt(result).reciprocal() * -2.5 + result.log2()
    result = result.sin()
    result.sum().backward()

    result_two = (inp > 0.5) * inp.exp2()
    result_two.sum().backward()

    # Mimic Receiver: print results
    assert_results(result, result_two, inp.grad)

if __name__ == "__main__":
    start = time()
    test_unary()
    end = time()
    
    print(f"Elapsed: {(end-start)*1000:.3f} ms")
