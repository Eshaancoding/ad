from ...node import Node
from autodiff import dot
from ..helper import indent

class DotProdNode (Node):
    def __init__(self, left:Node, right:Node):
        super().__init__([left, right])
        
        # assert shape
        left_shape = left.shape()
        right_shape = right.shape()
        
        assert len(left_shape) == 2, "Left shape of dot prod must be 2"
        assert len(right_shape) == 2, "Right shape of dot prod must be 2"
        
    def shape (self):
        # might need to be overriden by device or something
        return [self.left().shape()[0], self.right().shape()[1]] 
        
    def bck (self, grad:Node):
        self.left().bck(dot(grad, self.right().T()))
        self.right().bck(dot(self.left().T(), grad))
        
    def __repr__ (self):
        return f"Dot prod {self.left().shape()} x {self.right().shape()}\n{indent(self.left().__repr__())}\n{indent(self.right().__repr__())}"