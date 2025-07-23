from .. import Tensor
from typing import List

class SGD:
    def __init__(self, parameters: List[Tensor], lr=0.001):
        self.parameters = parameters
        self.lr = lr
    
    def step (self):
        for p in self.parameters:
            # this doesn't apply to it sadly; I have to figure something out in that case
            p -= self.lr * p.grad()
            