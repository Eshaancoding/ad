from ...node import Node
from copy import deepcopy
from .index import IndexNode

class ConcatNode (Node):
    __match_args__ = ("left", "right", "dim")
    def __init__(self, left:Node, right:Node, dim: int):
        
        # assert shape
        assert len(left.shape) == len(right.shape), "Concat shape is not equal"
        norm_dim = dim if dim >= 0 else len(left.shape) + dim
        for i in range(len(left.shape)):
            if i == norm_dim: continue
            assert left.shape[i] == right.shape[i], f"Concat dim mismatch: {left.shape} and {right.shape}"
        
        # calc shape
        d = deepcopy(left.shape)
        d[dim] += right.shape[dim]
        
        super().__init__([left, right], d)
        self.dim = dim
    
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
        return f"{self.id} = Concat at dim: {self.dim} --> ({self.left.id}, {self.right.id}) --> {self.children_datacmds}"