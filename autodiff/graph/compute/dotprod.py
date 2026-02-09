from ...node import Node
from colored import stylize, fore, style
import functools
    
class DotProdNode (Node):
    __match_args__ = ("left", "right")
    def __init__(self, left:Node, right:Node):
        # NOTE: If you have any other args in __match_args__, make sure to define them
        # assert shape
        assert len(left.shape) == 2, "Left shape of dot prod must be 2"
        assert len(right.shape) == 2, "Right shape of dot prod must be 2"

        assert left.shape[1] == right.shape[0], \
            f"Dot product dim mismatch. left: {left.shape} right: {right.shape}"
        
        super().__init__([left, right], shape=[left.shape[0], right.shape[1]])
        
    def _bck (self, grad:Node):
        return grad @ self.right.T(), self.left.T() @ grad
        
    def __repr__ (self):
        l_shape = self.children_shapes[0]
        r_shape = self.children_shapes[1]

        op_str = stylize("Dot Prod", fore("turquoise_2"))
        l_str = stylize(l_shape, fore("yellow") + style("bold"))
        r_str = stylize(r_shape, fore("yellow") + style("bold"))
        size_str = f"{l_str} x {r_str}"
        
        if self.kargs[0].is_none() or self.kargs[1].is_none() or self.kres.is_none():
            return f"{self.id} = {op_str} {size_str} --> ({self.left.id}, {self.right.id})"
        else:
            return f"{self.kres} = {op_str} {size_str} --> ({self.kargs[0]}, {self.kargs[1]})"
    
    def repeat_helper (self, is_child):
        if is_child:
            return (self.id,)
        else:
            return ("DotProd",)
