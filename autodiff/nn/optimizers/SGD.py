from .. import Node, Tensor
from typing import List

class SGD:
    def __init__(self, parameters: List[Node], lr=0.001):
        self.parameters = parameters
        self.lr = lr
    
    def step (self):
        for idx in range(len(self.parameters)):
            if isinstance(self.parameters[idx], Tensor):
                self.parameters[idx] -= self.lr * self.parameters[idx].grad()

    def zero_grad (self):
        for idx in range(len(self.parameters)):
            if isinstance(self.parameters[idx], Tensor):
                self.parameters[idx].grad_tensor = None
            
