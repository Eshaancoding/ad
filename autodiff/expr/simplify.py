from . import *
import math

# level 1 simplifications
def simpl_expr_inner (expr: Expression) -> Expression:
    if isinstance(expr, Value) or isinstance(expr, Val):
        return expr

    match expr:
        ############### Ones
        # _ / 1 --> _
        case Div(_ as node, Val(Constant(1))):
            return simpl_expr_inner(node)

        # _ * 1 --> _
        case Mult(_ as node, Val(Constant(1))):
            return simpl_expr_inner(node)
        
        # 1 * _ --> _
        case Mult(Val(Constant(1)), _ as node):
            return simpl_expr_inner(node)

        ############### Zeros
        # _ + 0 --> _
        case Add(_ as node, Val(Constant(0))):
            return simpl_expr_inner(node)
        
        # 0 + _ --> _
        case Add(Val(Constant(0)), _ as node):
            return simpl_expr_inner(node)

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


        ############### Bit simplifications
        # _ * 2^p --> _ << p
        case Mult(_ as node, Val(Constant(_ as constant))):
            l2 = math.log2(constant)
            if l2.is_integer():
                return simpl_expr_inner(ShiftLeft(node, Val(Constant(int(l2)))))

        # _ / 2^p --> _ >> p
        case Div(_ as node, Val(Constant(_ as constant))):
            l2 = math.log2(constant)
            if l2.is_integer():
                return simpl_expr_inner(ShiftRight(node, Val(Constant(int(l2)))))
            
        # _ % 2^p --> _ & (2^p - 1)
        case Remainder(_ as node, Val(Constant(_ as constant))):
            if math.log2(constant).is_integer():
                return simpl_expr_inner(BitwiseAnd(node, Val(Constant(constant - 1))))
        
        # ((_ % 3) % 3) --> _ % 3
        case Remainder(Remainder(_ as node, Val(Constant(_ as const_one))), Val(Constant(_ as const_two))):
            if const_one == const_two:
                return simpl_expr_inner(Remainder(node, Val(Constant(const_one))))

        # ((_ & 3) & 3) --> _ % 3
        case BitwiseAnd(BitwiseAnd(_ as node, Val(Constant(_ as const_one))), Val(Constant(_ as const_two))):
            if const_one == const_two:
                return simpl_expr_inner(BitwiseAnd(node, Val(Constant(const_one))))
        
        case _:
            pass
    
    if expr.is_binary():
        expr.a = simpl_expr_inner(expr.a)
        expr.b = simpl_expr_inner(expr.b)
        
    return expr

    
# this is kind of "brute force" method 
# didn't wanna think too hard...
# TODO: Might not scale well for large exprs
def simplify_expr (expr: Expression):
    start = expr
    while True:
        end = simpl_expr_inner(expr)
        
        if end == start: 
            break
        else:
            start = end
        
    return end