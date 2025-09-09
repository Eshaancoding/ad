import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autodiff.graph.data.receiver import Receiver
from autodiff.graph.tensor import Tensor
from autodiff.helper import benchmark
import autodiff
import numpy as np

def assert_results (result, result_two, input_grad):
    assert round(result.sum().item(), 5) == 6.2689
    assert round(result_two.sum().item(), 5) == 4.78375
    assert round(input_grad.sum().item(), 5) == 5.39896

def test_unary ():
    autodiff.graph.tensor.is_testing = True
    
    inp = Tensor(
        np.array([
            [0.3, 0.2, 0.7, 0.54],
            [0.13, 0.77, 0.23, 0.22]
        ]),
        requires_grad=True
    )

    result = inp.exp2() + 3
    result = result.sqrt().recip() * -2.5 + result.log2()
    result = result.sin()

    result_two = (inp > 0.5) * inp.exp2()
    autodiff.pg()
    result.backward()
    result_two.backward()

    Receiver(assert_results, [result, result_two, inp.grad()], name="saving params")
    benchmark(lambda: autodiff.execute(), name="full exec")

if __name__ == "__main__":
    test_unary()

"""
Equivalent test in Pytorch:

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
"""
