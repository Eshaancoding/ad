from autodiff.graph.data.receiver import Receiver
from autodiff.graph.tensor import Tensor
from autodiff.helper import benchmark
from autodiff.nn.normalization.layernorm import LayerNorm
import autodiff
import torch
import numpy as np

def test_variance ():
    # Tests sum
    torch.manual_seed(42)

    x = torch.tensor([[5.0, 6.0, 3.0], [4.0, 5.0, 6.0]])  # Shape: (2, 3)
    x = Tensor.from_torch(x, requires_grad=True)
    
    res = x.pow2()
    res.backward()

    def asrt (res, x_grad):
        assert np.all(res == np.array([
            [25, 36, 9],
            [16, 25, 36]
        ])).item() == True

        assert np.all(x_grad == np.array([
            [10, 12, 6],
            [8, 10, 12]
        ])).item() == True

    Receiver(asrt, [
        res,
        x.grad()
    ]);

    benchmark(lambda: autodiff.execute(), name="full exec")

if __name__ == "__main__":
    test_variance()


