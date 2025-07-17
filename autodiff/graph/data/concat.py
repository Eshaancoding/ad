from ...node import Node
from ..helper import indent
from copy import deepcopy
from .index import IndexNode

class ConcatNode (Node):
    __match_args__ = ("left", "right", "dim")
    def __init__(self, left:Node, right:Node, dim: int):
        super().__init__([left, right])
        
        # assert shape
        left_shape = left.shape
        right_shape = right.shape
        
        
        assert len(left_shape) == len(right_shape), "Concat shape is not equal"
        norm_dim = dim if dim >= 0 else len(left_shape) + dim
        for i in range(len(left_shape)):
            if i == norm_dim: continue
            assert left_shape[i] == right_shape[i], f"Concat dim mismatch: {left_shape} and {right_shape}"
        
        self.dim = dim
        self.left = left
        self.right = right
        
        # calc shape
        d = deepcopy(self.left.shape)
        d[self.dim] += self.right.shape[self.dim]
        self.shape = d
    
    def bck (self, grad:Node):
        l_dim = self.left.shape[self.dim]
        r_dim = self.right.shape[self.dim]

        self.left.bck(
            IndexNode(grad, 0, l_dim, self.dim)
        )
        
        self.right.bck(
            IndexNode(grad, l_dim, l_dim+r_dim, self.dim)
        )
    
    def __repr__ (self):
        return f"{self.id} = Concat at dim: {self.dim} --> ({self.left.id}, {self.right.id})"