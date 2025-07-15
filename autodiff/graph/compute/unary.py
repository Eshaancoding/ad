from typing import Dict, Callable, Optional, List
from ...node import Node
from ..helper import indent
from enum import Enum
from ...expr import NoneExpr
import math
from colored import stylize, fore, style

class UnaryOp (Enum):
    EXP2=1
    LOG2=2
    SIN=3
    RECIP=4
    SQRT=5
    EQUAL=6
    MORE_ZERO=7
    LESS_ZERO=8
    MORE_OR_EQ_ZERO=9
    LESS_OR_EQ_ZERO=10

class UnaryNode (Node):
    def __init__(self, child:Node, op: UnaryOp):
        super().__init__([child])

        self.shape = self.child().shape
        self.res_expr = NoneExpr()
        self.op = op
        
    def bck (self, grad:Node):
        if not isinstance(grad, Node):
            raise TypeError("Grad is not a node!")
        
        grad_dict: Dict[UnaryOp, Callable[[Node, Node], Optional[Node]]] = {
            # Derivatives
            UnaryOp.EXP2:  lambda grad, _: grad * math.log(2.0) * self,
            UnaryOp.LOG2:  lambda grad, parent: grad * math.log2(math.e) * parent.recip(),
            UnaryOp.SIN:   lambda grad, parent: grad * parent.cos(),
            UnaryOp.RECIP: lambda grad, parent: grad * (-1.0 / (parent.pow2())),
            UnaryOp.SQRT:  lambda grad, parent: grad * (1.0 / (2.0 * parent.sqrt())),
            
            # Comparative ops don't have gradient 
            UnaryOp.EQUAL: lambda _: None,
            UnaryOp.MORE_ZERO: lambda _: None,
            UnaryOp.LESS_ZERO: lambda _: None,
            UnaryOp.MORE_OR_EQ_ZERO: lambda _: None,
            UnaryOp.LESS_OR_EQ_ZERO: lambda _: None,
        }
        
        g = grad_dict[self.op](grad, self.child()) # in the backward sense, "child" becomes the "parent"
        if g is not None:
            self.child().bck(g)
        
    def __repr__(self) -> str:
        total = math.prod(self.shape)
        return f"{self.op} ({stylize(total, fore("yellow") + style("bold"))}) --> {self.res_expr}\n{indent(self.child().__repr__())}"