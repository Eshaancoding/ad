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
    
    def _bck (self, grad:Node):
        l_dim = self.left.shape[self.dim]
        r_dim = self.right.shape[self.dim]

        return IndexNode(grad, 0, l_dim, self.dim), \
               IndexNode(grad, l_dim, l_dim+r_dim, self.dim)
    
    def __repr__ (self):
        return f"{self.id} = Concat at dim: {self.dim} --> ({self.left.id}, {self.right.id})"

    def repeat_helper (self, is_child):
        return (
            "Concat",
            self.dim,
            self.left.repeat_helper(True) if is_child else (),
            self.right.repeat_helper(True) if is_child else ()
        )
