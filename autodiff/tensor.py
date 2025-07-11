from .node import Node
from typing import List
import random
import math

class Tensor (Node):
    def __init__(self, data: any):
        from autodiff import context
        super().__init__([])
        
        self.dim = list(Tensor._get_shape_and_validate(data))
        self.data = Tensor._flatten(data)
        self.grad = None
        
    def __init__ (self, data: List[float], dim: List[int]):
        from autodiff import context
        super().__init__([])

        self.dim = dim
        self.data = data
        self.grad = None

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
            
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad  
            
    def grad (self):
        if self.grad is None:
            raise RuntimeError("Gradient on tensor is none! Make sure you call .backward()")        

        return self.grad
    
    def shape (self) -> List[int]:
        return self.dim
    
    def __repr__ (self) -> str:
        return f"Tensor(id: {self.id}, dim: {self.dim})"
    
    def _gen_normal_random(mu=0, sigma=1):
        # Box-Muller transform
        u1 = random.random()
        u2 = random.random()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return mu + z0 * sigma
    
    def randn (shape:list[int]):
        shape = list(shape)
        return Tensor([Tensor._gen_normal_random() for _ in range(math.prod(shape))], shape)