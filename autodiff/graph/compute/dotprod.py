from ...node import Node
from autodiff import dot
from ..helper import indent
from ...expr import NoneExpr
from colored import stylize, fore, style

class DotProdNode (Node):
    def __init__(self, left:Node, right:Node):
        super().__init__([left, right])
        
        # assert shape
        left_shape = left.shape
        right_shape = right.shape
        
        # access expr 
        # must be shared + defined across nodes
        self.res_expr = NoneExpr()
        self.shape = [self.left().shape[0], self.right().shape[1]]
        
        assert len(left_shape) == 2, "Left shape of dot prod must be 2"
        assert len(right_shape) == 2, "Right shape of dot prod must be 2"
        
    def bck (self, grad:Node):
        self.left().bck(dot(grad, self.right().T()))
        self.right().bck(dot(self.left().T(), grad))
        
    def __repr__ (self):
        return f"{stylize(f"{self.inter_out} <-- ", fore("cyan")) if self.inter_out is not None else ""}Dot prod {stylize(self.left().shape, fore("yellow") + style("bold"))} x {stylize(self.right().shape, fore("yellow") + style("bold"))} --> {self.res_expr} \n{indent(self.left().__repr__())}\n{indent(self.right().__repr__())}"