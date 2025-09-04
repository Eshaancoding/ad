from ...node import Node
from enum import Enum
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
    __match_args__ = ("child", "op")
    def __init__(self, child:Node, op: UnaryOp):
        super().__init__([child], child.shape)
        
        self.op = op
        
    def _bck (self, grad:Node):
        if not isinstance(grad, Node):
            raise TypeError("Grad is not a node!")
        
        grad_dict = {
            # Derivatives
            UnaryOp.EXP2:  lambda grad, _: grad * math.log(2.0) * self,
            UnaryOp.LOG2:  lambda grad, parent: grad * math.log2(math.e) * parent.recip(),
            UnaryOp.SIN:   lambda grad, parent: grad * parent.cos(),
            UnaryOp.RECIP: lambda grad, parent: grad * (-1.0 / (parent.pow2())),
            UnaryOp.SQRT:  lambda grad, parent: grad * (1.0 / (2.0 * parent.sqrt())),
            
            # Comparative ops don't have gradient 
            UnaryOp.EQUAL: lambda _, _p: None,
            UnaryOp.MORE_ZERO: lambda _, _p: None,
            UnaryOp.LESS_ZERO: lambda _, _p: None,
            UnaryOp.MORE_OR_EQ_ZERO: lambda _, _p: None,
            UnaryOp.LESS_OR_EQ_ZERO: lambda _, _p: None,
        }
        
        return grad_dict[self.op](grad, self.child) # in the backward sense, "child" becomes the "parent"
        
    def __repr__ (self) -> str:
        total = math.prod(self.shape)
        op_str = stylize(f"{self.op.name}", fore("medium_turquoise"))
        size_str = stylize(str(total), fore("yellow") + style("bold"))
        if self.kargs[0].is_none() or self.kres.is_none():
            return f"{self.id} = {op_str} ({size_str}) --> ({self.child.id})"
        else:
            return f"{self.kres} = {op_str} ({size_str}) --> ({self.kargs[0]})"

    def repeat_helper (self, is_child):
        if is_child:
            return (self.id,)
        else:
            return ("Unary", self.op.value)
