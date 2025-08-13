from ...node import Node
from typing import List
import math

class ViewNode (Node):
    __match_args__ = ("child", "shape") 
    def __init__(self, child:Node, shape: list[int]):
        assert math.prod(child.shape) == math.prod(shape), "View dimensions are incorrect"
        super().__init__([child], shape)
        
    def handle_minus_dim(source_dim: List[int], input_dim: List[int]) -> List[int]:
        if -1 in input_dim:
            idx = input_dim.index(-1)
            
            total_size = 1
            for d in source_dim:
                total_size *= d
            
            input_size = 1
            for d in input_dim:
                if d != -1:
                    input_size *= d

            assert total_size % input_size == 0, "Can't fill in -1"
            inferred_dim = total_size // input_size

            # Replace -1 with inferred value
            input_dim = input_dim.copy()
            input_dim[idx] = inferred_dim

        return list(input_dim)
        
    def bck(self, grad: Node):
        if not isinstance(grad, Node):
            raise TypeError("Grad is not node!")
        
        self.child.bck(grad.view(self.children_shapes[0]))

    def __repr__ (self):
        return f"{self.id} = View from {self.children_shapes[0]} to {self.shape} --> ({self.child.id})"

    def node_eq(self, other) -> bool:
        if not isinstance(other, ViewNode):
            return False

        return \
            self.shape == other.shape and \
            self.child.node_eq(other.child)
