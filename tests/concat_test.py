from autodiff.graph.data.receiver import Receiver
from autodiff.graph.tensor import Tensor
from autodiff.helper import benchmark
import autodiff
import torch
import numpy as np
np.set_printoptions(suppress=True, precision=4)

def concat_test ():
    # Tests sum
    torch.manual_seed(42)

    x = torch.tensor([
        [5.0, 6.0, 3.0, 1.0], 
        [4.0, 5.0, 6.0, 3.0]
    ])  # Shape: (2, 4)
    x = Tensor.from_torch(x, requires_grad=True)

    y = torch.tensor([
        [8.0, 9.0, 3.0, 6.0], 
        [3.0, 5.0, 4.0, 2.0]
    ])  # Shape: (2, 4)
    y = Tensor.from_torch(y, requires_grad=True)
   
    res = autodiff.concat([x*2, y*3], 1).contigious()
    res.backward() 

    def asrt (res, x_grad, y_grad):
        assert x_grad.sum() == 16, "X grad wrong"
        assert y_grad.sum() == 24, "y_grad wrong"
        assert np.all(res == np.array([
            [10, 12,  6,  2, 24, 27,  9, 18],
            [ 8, 10, 12,  6,  9, 15, 12,  6]
        ])).item()


    Receiver(asrt, [ res, x.grad(), y.grad() ]);

    benchmark(lambda: autodiff.execute(), name="full exec")

if __name__ == "__main__":
    concat_test()
