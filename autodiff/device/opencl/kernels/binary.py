from ....graph import *
from ....context import *
from ..context import ADCLContext
from ..karg import *
from math import prod

def lower_binary (cmd: BinaryNode):
    op_str = ""
    match cmd.op:
        case BinaryOp.ADD:
            op_str = "+"
        case BinaryOp.MULT:
            op_str = "*"

    return f"{lower_karg(cmd.kres)} = {lower_karg(cmd.kargs[0])} {op_str} {lower_karg(cmd.kargs[1])};"

def execute_binary (context: ADCLContext, cmd: BinaryNode):
    name = f"binary_{context.get_prog_id()}"
    args, program_args = lower_args(cmd)

    # construct program
    program = context.get_program(name, f"""
__kernel void {name} (
    {program_args}
) {{
    const size_t _global_id = get_global_id(0);
    {lower_binary(cmd)} 
}}           
    """.strip())
    
    # buffer args
    args = [context.get_buffer(buf_id) for buf_id in args]
   
    program(context.command_queue, (prod(cmd.shape), ), None, *args) 