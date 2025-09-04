from ..node import Node
from typing import List, Optional
import random
import math

# Numpy is the only **major** dependency of this tensor library
# Numpy handles printing, shape, data types, etc. that I do not want to implement
# Easy interface to cltypes as well (which is used in backend)
import numpy as np

class Tensor (Node):
    __match_args__ = ("arr")
    def __init__ (self, arr: np.array, requires_grad:bool=False):
        super().__init__([], list(arr.shape))

        self.arr = arr.astype(np.float32) # ensure float32 (might make a copy)
        self.grad_tensor = None
        self.requires_grad = requires_grad

    def _bck (self, grad):
        if not isinstance(grad, Node):
            raise TypeError("Backward on tensor grad is not tensoor")
            
        if self.requires_grad: # never calculate gradient unnecessarily
            if self.grad_tensor is None:
                self.grad_tensor = grad
            else:
                self.grad_tensor += grad 

        return None
            
    def grad (self):
        if self.grad_tensor is None:
            raise RuntimeError(f"Gradient on tensor {self.id} is none! Make sure you call .backward()")        

        return self.grad_tensor

    def repeat_helper(self, is_child):
        return (
            "Tensor",
            self.id
        )
    
    def __repr__ (self) -> str:
        return f"Tensor(id: {self.id}, orig_shape: {self.shape})"

    @staticmethod
    def randn (shape:list[int] | tuple):
        return Tensor(np.random.randn(*shape))
    
    @staticmethod
    def fill (val: float, shape:list[int]):
        return Tensor(np.full(tuple(shape), val))
    
    @staticmethod
    def ones (shape: list[int]):
        return Tensor.fill(1.0, shape)
    
    @staticmethod
    def zeros (shape: list[int]):
        return Tensor.fill(0.0, shape)
