from ...node import Node
import math
from enum import Enum
from colored import stylize, fore, style
import functools

class BinaryOp (Enum):
    ADD=1
    MULT=2

class BinaryNode (Node):
    __match_args__ = ("left", "right", "op", "in_place")
    def __init__(self, left: Node, right: Node, op: BinaryOp, in_place:bool = False):
        assert left.shape == right.shape, f"Dimensional mismatch at binary! {left.shape}: {left} and {right.shape}: {right}"

        super().__init__([left, right], left.shape, None)
        self.op = op
        self.in_place = in_place
    
    def _bck (self, grad:Node):
        if not isinstance(grad, Node):
            raise TypeError("Grad is not a node!")
        
        if self.op == BinaryOp.ADD:
            return grad, grad
        elif self.op == BinaryOp.MULT:
            return grad * self.right, grad * self.left

    def __repr__ (self) -> str:
        total = math.prod(self.shape)
        in_place_str = " IN PLACE" if self.in_place else ""
        op_str = stylize(f"{self.op.name}{in_place_str}", fore("medium_turquoise"))
        size_str = stylize(str(total), fore("yellow") + style("bold"))
        
        if self.kargs[0].is_none() or self.kargs[1].is_none() or self.kres.is_none():
            return f"{self.id} = {op_str} ({size_str}) --> ({self.left.id}, {self.right.id})"
        else:
            return f"{self.kres} = {op_str} ({size_str}) --> ({self.kargs[0]}, {self.kargs[1]})"
    
    def repeat_helper(self, is_child):
        if is_child:
            return (self.id,)
        else:
            return (
                "Binary",
                self.op.value,
                self.in_place
            )
