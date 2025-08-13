from autodiff import execute
from autodiff.graph.data.receiver import Receiver
from autodiff.graph.tensor import Tensor
import numpy as np

from autodiff.nn.loss.categorical import categorical_cross_entropy

a = Tensor(np.array([[0.3, 0.6, 0.9]]))
b = Tensor(np.array([[1, 0, 0]]))

loss = categorical_cross_entropy(a, b)
loss.backward()

def print_loss (l):
    print(l)

Receiver(print_loss, [loss])

execute()
