from ...node import Node
from ..helper import indent
from enum import Enum
from typing import List

class BinaryOp (Enum):
    ADD=1
    MULT=2

class BinaryNode (Node):
    def __init__(self, left: Node, right: Node, op: BinaryOp):
        super().__init__([left, right])

        assert left.shape() == right.shape(), f"Dimensional mismatch at binary! {left.shape()} and {right.shape()}"
        self.op = op
    
    def bck (self, grad:Node):
        if not isinstance(grad, Node):
            raise TypeError("Grad is not a node!")
        
        if self.op == BinaryOp.ADD:
            self.left().bck(grad)
            self.right().bck(grad)
        elif self.op == BinaryOp.MULT:
            self.left().bck(grad * self.right())
            self.right().bck(grad * self.left())

    def shape (self) -> List[int]:
        return self.left().shape()
    
    def __repr__ (self) -> str:
        return f"{self.op}\n{indent(self.left().__repr__())}\n{indent(self.right().__repr__())}"