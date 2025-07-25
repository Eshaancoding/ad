from ....graph import *
from ....context import *
from ....fusion import *
from ....alloc import *
from ..context import ADCLContext
from ..karg import *
from math import prod

from .binary import lower_binary
from .unary import lower_unary

def lower_elwfuse (fused_cmd: ElwFuse):
    res = ""
    for cmd in fused_cmd.nodes:
        match cmd:
            case BinaryNode():
                res += lower_binary(cmd)
            case UnaryNode():
                res += lower_unary(cmd)
            case AllocEntry():
                res += lower_temp_alloc(cmd)
        res += "\n    "
    return res        
         
def execute_elwfuse (context: ADCLContext, cmd: ElwFuse):
    name = f"elwfuse_{context.get_prog_id()}"
    args, program_args = lower_args(cmd)

    # construct program
    program = context.get_program(name, f"""
__kernel void {name} (
    {program_args}
) {{
    const size_t _global_id = get_global_id(0);
    {lower_elwfuse(cmd)} 
}}           
    """.strip())
    
    # buffer args
    args = [context.get_buffer(buf_id) for buf_id in args]
    program(context.command_queue, (prod(cmd.get_elw().shape), ), None, *args) 