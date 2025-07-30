from ..node import Node
from typing import List, Optional
import random
import math

class Tensor (Node):
    __match_args__ = ("data", "shape")
    def __init__(self, data, shape: Optional[List[int]]):
        # Tensor 
        
        sh = None
        if shape is not None:
            sh = shape 
            self.data = data 
        else: 
            sh = list(Tensor._get_shape_and_validate(data))
            self.data = Tensor._flatten(data)

        super().__init__([], sh)

        from ..context import context  

        # always add dep list of this
        context.add_dep_list(self.id) 
        self.grad_tensor = None 

    @staticmethod
    def _get_shape_and_validate(data):
        if not isinstance(data, list):
            return ()  # Base case: leaf node (e.g., float)

        # Get the expected length from the first sub-element
        length = len(data)
        sub_shapes = []

        for item in data:
            if not isinstance(item, list):
                if item != data[0]:  # Inconsistent types
                    raise ValueError("Inconsistent tensor: mixed types at a level")
                return (length,)  # Flat list of scalars

            sub_shape = Tensor._get_shape_and_validate(item)
            sub_shapes.append(sub_shape)

        # Check all sub-shapes are equal
        first_shape = sub_shapes[0]
        for s in sub_shapes:
            if s != first_shape:
                raise ValueError(f"Mismatched sub-shapes found: {sub_shapes}")

        return (length,) + first_shape
    
    @staticmethod
    def _flatten (tensor):
        if not isinstance(tensor, list):
            return [tensor]
        else:
            flat_list = []
            for item in tensor:
                flat_list.extend(Tensor._flatten(item))
            return flat_list
        
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
            raise RuntimeError("Gradient on tensor is none! Make sure you call .backward()")        

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
    def _gen_normal_random(mu=0, sigma=1):
        # Box-Muller transform
        u1 = random.random()
        u2 = random.random()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return mu + z0 * sigma
    
    @staticmethod
    def randn (shape:list[int] | tuple):
        shape = list(shape)
        return Tensor([Tensor._gen_normal_random() for _ in range(math.prod(shape))], shape)
    
    @staticmethod
    def fill (val: float, shape:list[int]):
        shape = list(shape)
        return Tensor([val for _ in range(math.prod(shape))], shape)
    
    @staticmethod
    def ones (shape: list[int]):
        return Tensor.fill(1.0, shape)
    
    @staticmethod
    def zeros (shape: list[int]):
        return Tensor.fill(0.0, shape)
