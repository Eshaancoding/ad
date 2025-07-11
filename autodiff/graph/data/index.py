from ...node import Node
from ..helper import indent
from copy import deepcopy
from .constant import ConstantNode

class IndexNode (Node):
    def __init__(self, child:Node, start:int, end:int, dim:int):
        super().__init__([child])
        
        self.start = start
        self.end = end
        self.dim = dim
        
    def shape (self):
        d = deepcopy(self.child().shape()) 
        d[self.dim] = self.end - self.start
        return d
    
    def bck(self, grad):
        c_dim = self.child().shape()        
        
        first_dim = deepcopy(c_dim)
        first_dim[self.dim] = self.start
        
        second_dim = deepcopy(c_dim)
        second_dim[self.dim] = c_dim[self.dim] - self.end
        
        first = ConstantNode(0, first_dim)
        second = ConstantNode(0, second_dim)
        
        from autodiff import concat
        g = concat([first, grad, second], self.dim)
        self.child().bck(g)
        
    def __repr__ (self):
        return f"Index dim: {self.dim} from {self.start} to {self.end}\n{indent(self.child().__repr__())}"