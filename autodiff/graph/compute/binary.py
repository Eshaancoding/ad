from autodiff.context import context
from autodiff.graph.tensor import Tensor
from ...node import Node
import math
from enum import Enum
from ...expr import NoneExpr
from colored import stylize, fore, style

class BinaryOp (Enum):
    ADD=1
    MULT=2

class BinaryNode (Node):
    __match_args__ = ("left", "right", "op", "in_place")
    def __init__(self, left: Node, right: Node, op: BinaryOp, in_place:bool = False):
        assert left.shape == right.shape, f"Dimensional mismatch at binary! {left.shape}: {left} and {right.shape}: {right}"

        super().__init__([left, right], left.shape, left.id if in_place else None)
        self.op = op
        self.in_place = in_place

        if self.in_place and isinstance(left, Tensor): # if in place, add to dependency list
            context.add_dep_list(left)
    
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
        if self.kargs[0].is_none() or self.kargs[1].is_none() or self.kres.is_none():
            return f"{self.id} = {self.op} ({stylize(str(total), fore("yellow") + style("bold"))}) --> ({self.left.id}, {self.right.id})"
        else:
            return f"{self.kres} = {self.op} ({stylize(str(total), fore("yellow") + style("bold"))}) --> ({self.kargs[0]}, {self.kargs[1]})"
    
    def node_eq (self, other:Node) -> bool:
        if not isinstance(other, BinaryNode):
            return False
        
        return \
            self.op == other.op and \
            self.in_place == other.in_place and \
            self.left.node_eq(other.left) and \
            self.right.node_eq(other.right)
            
