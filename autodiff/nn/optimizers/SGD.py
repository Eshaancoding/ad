from .. import Node, Tensor
from typing import List

class SGD:
    def __init__(self, parameters: List[Node], lr=0.001):
        self.parameters = parameters
        self.lr = lr
    
    def step (self):
        for p in self.parameters:
            if isinstance(p, Tensor):
                # this doesn't apply to it sadly; I have to figure something out in that case
                p -= self.lr * p.grad()

    def zero_grad (self):
        for p in self.parameters:
            if isinstance(p, Tensor):
                p.grad_tensor = None
            
