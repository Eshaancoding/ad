from ...node import Node
import math
from ..helper import indent
from enum import Enum
from typing import List
from ...expr import NoneExpr
from colored import stylize, fore, style

class BinaryOp (Enum):
    ADD=1
    MULT=2

class BinaryNode (Node):
    def __init__(self, left: Node, right: Node, op: BinaryOp):
        super().__init__([left, right])

        assert left.shape == right.shape, f"Dimensional mismatch at binary! {left.shape} and {right.shape}"
        self.op = op
        
        # must be shared + defined across nodes (TODO: move to super().__init__())
        self.res_expr = NoneExpr()
        self.shape = self.left().shape 
    
    def bck (self, grad:Node):
        if not isinstance(grad, Node):
            raise TypeError("Grad is not a node!")
        
        if self.op == BinaryOp.ADD:
            self.left().bck(grad)
            self.right().bck(grad)
        elif self.op == BinaryOp.MULT:
            self.left().bck(grad * self.right())
            self.right().bck(grad * self.left())

    def __repr__ (self) -> str:
        total = math.prod(self.shape)
        return f"{self.op} ({stylize(total, fore("yellow") + style("bold"))}) --> {self.res_expr}\n{indent(self.left().__repr__())}\n{indent(self.right().__repr__())}"