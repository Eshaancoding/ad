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
    def __init__ (self, arr: np.array):
        super().__init__([], list(arr.shape))

        self.arr = arr.astype(np.float32) # ensure float32

        # always add dep list of this
        from ..context import context  
        context.add_dep_list(self.id) 
        self.grad_tensor = None

    def bck (self, grad):
        if not isinstance(grad, Node):
            raise TypeError("Backward on tensor grad is not tensoor")
            
        if self.grad_tensor is None:
            self.grad_tensor = grad

            # add to grad to dep list only if lenient_dep
            from ..context import context  
            if context.lenient_dep:
                context.add_dep_list(self.grad_tensor.id)
        else:
            self.grad_tensor += grad 
            
    def grad (self):
        if self.grad_tensor is None:
            raise RuntimeError(f"Gradient on tensor {self.id} is none! Make sure you call .backward()")        

        return self.grad_tensor

    def node_eq(self, other) -> bool:
        if not isinstance(other, Tensor):
            return False

        # only case where I am not declaring same data and shape, and using id
        # if the user declares the same shape and same data, then likely the user is using it for different purposes
        return self.id == other.id 
    
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
