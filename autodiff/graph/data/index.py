from ...node import Node
from copy import deepcopy
from .constant import ConstantNode
from ...expr import NoneExpr

class IndexNode (Node):
    __match_args__ = ("child", "start", "end", "dim")
    def __init__(self, child:Node, start:int, end:int, dim:int):
        # calc shape 
        d = deepcopy(child.shape) 
        d[dim] = end - start
    
        # init
        super().__init__([child], d)
        self.start = start
        self.end = end
        self.dim = dim
            
    def bck(self, grad):
        c_dim = self.children_shapes[0]
        
        first_dim = deepcopy(c_dim)
        first_dim[self.dim] = self.start
        
        second_dim = deepcopy(c_dim)
        second_dim[self.dim] = c_dim[self.dim] - self.end
        
        first = ConstantNode(0, first_dim)
        second = ConstantNode(0, second_dim)
        
        from autodiff import concat
        g = concat([first, grad, second], self.dim)
        self.child.bck(g)
        
    def __repr__ (self):
        return f"{self.id} = Index dim: {self.dim} from {self.start} to {self.end} --> ({self.child.id})"

    def node_eq(self, other) -> bool:
        if not isinstance(other, IndexNode):
            return False

        return \
            self.start == other.start and \
            self.end == other.end and \
            self.dim == other.dim and \
            self.child.node_eq(other.child)
