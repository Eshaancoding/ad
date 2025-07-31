from ....graph import *
from ....context import *
from ..context import ADCLContext
from ..karg import *
from math import prod

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

def execute_unary (context: ADCLContext, cmd: UnaryNode):
    name = f"unary_{cmd.program_id}"
    args, program_args = lower_args(cmd)

    # construct program
    program = context.get_program(name, f"""
__kernel void {name} (
    {program_args}
) {{
    const size_t _global_id = get_global_id(0);
    {lower_unary(cmd)} 
}}           
    """.strip())
    
    # buffer args
    args = [context.get_buffer(buf_id) for buf_id in args]
   
    program(
        context.command_queue,   # Command queue
        (prod(cmd.shape), ),     # global size 
        None,                    # local size
        *args                    # arguments
    )
