from ....graph import *
from ....context import *
from ..karg import *
from math import prod
from ..cuda_helper import *
from ..device import CudaDevice

def lower_unary (cmd: UnaryNode):
    expr = ""
    arg = lower_karg(cmd.kargs[0])

    # you can experiment with fast math!
    match cmd.op:
        case UnaryOp.EXP2:
            expr = f"exp2f({arg})" 
        case UnaryOp.LOG2:
            expr = f"log2f({arg})"
        case UnaryOp.SIN:
            expr = f"sinf({arg})"
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

def init_unary (dev: CudaDevice, cmd: UnaryNode):
    name = f"unary_{cmd.program_id}"
    args, program_args = lower_args(cmd)

    # construct program
    program_str = f"""
__global__ void {name} (
    {program_args}
) {{
    int _global_id = blockIdx.x;
    {lower_unary(cmd)} 
}}           
    """.strip()
    
    # buffer args
    args = [Buffer(dev.buffers[buf_id]) for buf_id in args]

    return build_kernel(
        name, 
        program_str, 
        args, 
        dev.arch,
        (prod(cmd.shape), ), 
        None
    )
