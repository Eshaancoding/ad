# Lowering kargs
from ...expr import *
from ...kernalize import *
from ...node import Node

def lower_expr (expr: Expression | Value) -> str:
    match expr:
        case Constant(val):
            return str(val)
        case Global():
            return "_global_id"
        case X():
            return "_x"
        case Y():
            return "_y"
        case Val(v):
            return lower_expr(v)
        case Add(a, b):
            return f"({lower_expr(a)} + {lower_expr(b)})"
        case Minus(a, b):
            return f"({lower_expr(a)} + {lower_expr(b)})"
        case Mult(a, b):
            return f"({lower_expr(a)} * {lower_expr(b)})"
        case Div(a, b):
            return f"({lower_expr(a)} / {lower_expr(b)})"
        case Remainder(a, b):
            return f"({lower_expr(a)} % {lower_expr(b)})"
        case ShiftRight(a, b):
            return f"({lower_expr(a)} >> {lower_expr(b)})"
        case ShiftLeft(a, b):
            return f"({lower_expr(a)} << {lower_expr(b)})"
        case BitwiseAnd(a, b):
            return f"({lower_expr(a)} & {lower_expr(b)})"
        case MoreThan(a, b):
            return f"({lower_expr(a)} > {lower_expr(b)})"
        case LessThan(a, b):
            return f"({lower_expr(a)} < {lower_expr(b)})"
        case NoneExpr():
            raise Exception("Encountered NoneExpr at lowering karg in device")
        case _:
            return Exception(f"Invalid expr at lower: {expr}")
        
def lower_karg (karg: KernelArg):
    if isinstance(karg, KMatrix):
        return f"{karg.id}[{lower_expr(karg.access)}]"
    elif isinstance(karg, KConcat):
        return f"({lower_expr(karg.condition)} ? {lower_karg(karg.karg_one)} : {lower_karg(karg.karg_two)})"
    elif isinstance(karg, KConstant):
        return str(karg.constant)
    
def lower_args (node: Node):
    args = node.kargs_child_ids()
    args = [f"__global float* {arg_id}" for arg_id in args] 
    return ", ".join(args)