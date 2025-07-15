from ...node import Node
from ..helper import indent
from copy import deepcopy
from .index import IndexNode

class ConcatNode (Node):
    def __init__(self, left:Node, right:Node, dim: int):
        super().__init__([left, right])
        
        # assert shape
        left_shape = left.shape
        right_shape = right.shape
        
        assert len(left_shape) == len(right_shape), "Concat shape is not equal"
        for i in range(len(left_shape)):
            if i == dim: continue
            assert left_shape[i] == right_shape[i], "Concat dim mismatch"
        
        self.dim = dim
        
        # calc shape
        d = deepcopy(self.left().shape)
        d[self.dim] += self.right().shape[self.dim]
        self.shape = d
    
    def bck (self, grad:Node):
        l_dim = self.left().shape[self.dim]
        r_dim = self.right().shape[self.dim]

        self.left().bck(
            IndexNode(grad, 0, l_dim, self.dim)
        )
        
        self.right().bck(
            IndexNode(grad, l_dim, l_dim+r_dim, self.dim)
        )
    
    def __repr__ (self):
        return f"Concat at dim: {self.dim}\n{indent(self.left().__repr__())}\n{indent(self.right().__repr__())}"