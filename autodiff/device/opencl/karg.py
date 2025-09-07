# Lowering kargs
from ...expr import *
from ...kernalize import *
from ...node import Node
from ...alloc import *
from ...fusion import *

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
            return f"({lower_expr(a)} - {lower_expr(b)})"
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
        if karg.is_temp:
            return f"temp_{karg.id}"
        else:
            return f"buf_{karg.id}[{lower_expr(karg.access)}]"
    elif isinstance(karg, KConcat):
        return f"({lower_expr(karg.condition)} ? ({lower_karg(karg.karg_two)}) : ({lower_karg(karg.karg_one)}))"
    elif isinstance(karg, KConstant):
        return str(karg.constant)
    
def lower_args (node: Node | FuseBase):
    args = []
    if isinstance(node, Node):
        args = node.kargs_child_ids(filter_temp=True)

        # append res args as well
        ids = node.kres.get_ids(filter_temp=True)
        if len(ids) == 1:
            res_id = list(ids)[0]
            if res_id not in args:
                args.append(res_id)
        elif len(ids) != 0:
            raise Exception("Invalid len of result ids")
    elif isinstance(node, FuseBase):
        for n in node.nodes:
            if isinstance(n, Node):
                # child id
                args.extend(n.kargs_child_ids(filter_temp=True))
                
                # add result
                ids = n.kres.get_ids(filter_temp=True)
                if len(ids) == 1:
                    res_id = list(ids)[0]
                    args.append(res_id)
                elif len(ids) != 0:
                    raise Exception("Invalid len of result ids")
        args = list(set(args))
        
    program_args = [f"__global float* buf_{arg_id}" for arg_id in args] 
    return args, ",\n    ".join(program_args)

def lower_temp_alloc (alloc_entry: AllocEntry):
    assert alloc_entry.is_temp, "Alloc entry not temp at in fused op!"
    return f"float temp_{alloc_entry.id} = 0.0;"
