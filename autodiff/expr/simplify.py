from . import *
from copy import deepcopy
import math
from typing import Optional

def simpl_expr_inner (expr: Expression, size: Optional[int]) -> Expression:
    if isinstance(expr, Value) or isinstance(expr, Val):
        return expr

    match expr:
        ############### Constant Simplification

        ############### Ones
        # _ / 1 --> _
        case Div(_ as node, Val(Constant(1))):
            return simpl_expr_inner(node, size)

        # _ * 1 --> _
        case Mult(_ as node, Val(Constant(1))):
            return simpl_expr_inner(node, size)
        

        # 1 * _ --> _
        case Mult(Val(Constant(1)), _ as node):
            return simpl_expr_inner(node, size)

        ############### Zeros
        # _ + 0 --> _
        case Add(_ as node, Val(Constant(0))):
            return simpl_expr_inner(node, size)
        
        # 0 + _ --> _
        case Add(Val(Constant(0)), _ as node):
            return simpl_expr_inner(node, size)

        # _ * 0 --> 0
        case Mult(_, Val(Constant(0))):
            return Val(Constant(0))
        
        # 0 * _ --> 0
        case Mult(Val(Constant(0)), _):
            return Val(Constant(0))
        
        # _ & 0 --> 0
        case BitwiseAnd(_, Val(Constant(0))):
            return Val(Constant(0))
        
        # 0 << _ = 0
        case ShiftLeft(Val(Constant(0)), _):
            return Val(Constant(0))
        
        # 0 >> _ = 0
        case ShiftRight(Val(Constant(0)), _):
            return Val(Constant(0))

        ############### Conatenation Simplification
        case Minus(Minus(_ as g, Val(Constant(n1))), Val(Constant(n2))):
            return simpl_expr_inner(Minus(g, Val(Constant(n1 + n2))), size)
        
        case MoreThan(Minus(_ as g, Val(Constant(n1))), Val(Constant(n2))):
            return simpl_expr_inner(MoreThan(g, Val(Constant(n1 + n2))), size)

        ############### Bit simplifications
        # _ * 2^p --> _ << p
        case Mult(_ as node, Val(Constant(_ as constant))):
            l2 = math.log2(constant)
            if l2.is_integer():
                return simpl_expr_inner(ShiftLeft(node, Val(Constant(int(l2)))), size)

        # _ / 2^p --> _ >> p
        case Div(_ as node, Val(Constant(_ as constant))):
            l2 = math.log2(constant)
            if l2.is_integer():
                return simpl_expr_inner(ShiftRight(node, Val(Constant(int(l2)))), size)
            
        # _ % 2^p --> _ & (2^p - 1)
        case Remainder(_ as node, Val(Constant(_ as constant))):
            if math.log2(constant).is_integer():
                return simpl_expr_inner(BitwiseAnd(node, Val(Constant(constant - 1))), size)
        
        # ((_ % 3) % 3) --> _ % 3
        case Remainder(Remainder(_ as node, Val(Constant(_ as const_one))), Val(Constant(_ as const_two))):
            if const_one == const_two:
                return simpl_expr_inner(Remainder(node, Val(Constant(const_one))), size)

        # ((_ & 3) & 3) --> _ % 3
        case BitwiseAnd(BitwiseAnd(_ as node, Val(Constant(_ as const_one))), Val(Constant(_ as const_two))):
            if const_one == const_two:
                return simpl_expr_inner(BitwiseAnd(node, Val(Constant(const_one))), size)
        
        # ((_ >> v) & b << v) + _ & a
        # if log2(b+1) and log2(a+1) are integers then...
        # ((_ >> v) & b << v) + _ & a <-- _ % (2^(log2(b+1) + log2(a+1)))
        # There exists a version where it's dividing and multiplying
        case Add(
            ShiftLeft(
                BitwiseAnd(
                    ShiftRight(
                        _ as gl,
                        Val(Constant(_ as v1))
                    ),
                    Val(Constant(_ as b))   
                ),
                Val(Constant(_ as v))    
            ),
            BitwiseAnd(
                _ as gl2,
                Val(Constant(_ as a))
            )
        ):
            if gl == gl2 and v == v1 and math.log2(b+1).is_integer() and math.log2(a+1).is_integer():
                res_remainder = int(pow(2, math.log2(b+1) + math.log2(a+1)))
                return simpl_expr_inner(Remainder(gl, Val(Constant(res_remainder))), size) 
        
        # If the size of the command is given, then we can do a simple simplification
        # 144 = UnaryOp.EXP2 (16) (Mat (id: 143, access: (Global % 16)))
        # Since the size of global IS 16, then we know that will never reach above >16 and &16 gaurd is useless
        case Remainder(_ as val, Val(Constant(_ as s))) if size is not None:
            if s == size:
                return simpl_expr_inner(val, size)

        # If the size of the command is given, then we can do a simple simplification
        # ((Global >> 7) & 3) when we know size is 512
        # In this case, we can simplify this expr to Global >> 7, as we know the max is 511, and 511 >> 7 is always <= 3
        # If the size of the command is given, then we can do a simple simplification
        # 144 = UnaryOp.EXP2 (16) (Mat (id: 143, access: (Global & 15)))
        # Since the size of global IS 16, then we know that will never reach above >16 and &16 gaurd is useless
        case BitwiseAnd(_ as val, Val(Constant(_ as s))) if size is not None:
            if s + 1 == size:
                return simpl_expr_inner(val, size) 
            else:
                # what if val must be simplified in order to test out bitwise and? 
                # for example, >> (val) is common 
                # TODO: Abstract this to any operation?
                match val:
                    case ShiftRight(Global(), Val(Constant(_ as c))):
                        if ((size-1) >> c) <= s:
                            return simpl_expr_inner(val, size) 
        case _:
            pass
        
    if expr.is_binary():
        expr.a = simpl_expr_inner(expr.a, size)
        expr.b = simpl_expr_inner(expr.b, size)
        
    return expr
    
# this is kind of "brute force" method 
# didn't wanna think too hard...
# TODO: Might not scale well for large exprs
def simplify_expr (expr: Expression, size: Optional[int]):
    # return expr
    start = expr
    while True:
        end = simpl_expr_inner(deepcopy(start), size)
        
        if end == start: 
            break
        else:
            start = end
        
    return end
