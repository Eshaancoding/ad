from ...node import Node
import math
from enum import Enum
from ...expr import NoneExpr
from colored import stylize, fore, style

class BinaryOp (Enum):
    ADD=1
    MULT=2

class BinaryNode (Node):
    __match_args__ = ("left", "right", "op")
    def __init__(self, left: Node, right: Node, op: BinaryOp):
        assert left.shape == right.shape, f"Dimensional mismatch at binary! {left.shape}: {left} and {right.shape}: {right}"

        super().__init__([left, right], left.shape)
        self.op = op
    
    def bck (self, grad:Node):
        if not isinstance(grad, Node):
            raise TypeError("Grad is not a node!")
        
        if self.op == BinaryOp.ADD:
            self.left.bck(grad)
            self.right.bck(grad)
        elif self.op == BinaryOp.MULT:
            self.left.bck(grad * self.right)
            self.right.bck(grad * self.left)

    def __repr__ (self) -> str:
        total = math.prod(self.shape)
        if self.kargs[0].is_none() or self.kargs[1].is_none():
            return f"{self.id} = {self.op} ({stylize(total, fore("yellow") + style("bold"))}) --> ({self.left.id}, {self.right.id})"
        else:
            return f"{self.id} = {self.op} ({stylize(total, fore("yellow") + style("bold"))}) --> ({self.kargs[0]}, {self.kargs[1]})"