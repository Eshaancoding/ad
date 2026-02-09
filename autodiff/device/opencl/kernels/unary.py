from ....graph import *
from ....context import *
from ..karg import *
from math import prod
from ..cl_helper import *
from ..device import OpenCLDevice

def lower_unary (cmd: UnaryNode):
    expr = ""
    arg = lower_karg(cmd.kargs[0])
    match cmd.op:
        case UnaryOp.EXP2:
            expr = f"exp2({arg})"
        case UnaryOp.LOG2:
            expr = f"log2({arg})"
        case UnaryOp.SIN:
            expr = f"sin({arg})"
        case UnaryOp.RECIP:
            expr = f"1.0/{arg}"
        case UnaryOp.SQRT:
            expr = f"sqrt(fabs({arg}))" 
        case UnaryOp.EQUAL:
            expr = f"({arg} == 0.0f) ? 1.0 : 0.0"
        case UnaryOp.MORE_ZERO:
            expr = f"({arg} > 0.0f) ? 1.0 : 0.0"
        case UnaryOp.LESS_ZERO:
            expr = f"({arg} < 0.0f) ? 1.0 : 0.0"
        case UnaryOp.MORE_OR_EQ_ZERO:
            expr = f"({arg} >= 0.0f) ? 1.0 : 0.0"
        case UnaryOp.LESS_OR_EQ_ZERO:
            expr = f"({arg} <= 0.0f) ? 1.0 : 0.0"

    return f"{lower_karg(cmd.kres)} = {expr};"

def init_unary (dev: OpenCLDevice, cmd: UnaryNode):
    name = f"unary_{cmd.program_id}"
    args, program_args = lower_args(cmd)

    # construct program
    program_str = f"""
__kernel void {name} (
    {program_args}
) {{
    const size_t _global_id = get_global_id(0);
    {lower_unary(cmd)} 
}}           
    """.strip()
    
    # buffer args
    args = [Buffer(dev.buffers[buf_id]) for buf_id in args]

    return build_kernel(dev, name, program_str, args, (prod(cmd.shape), ), None)
