from ....graph import *
from ....context import *
from ..context import ADCLContext
from ..karg import *
from math import prod

# Apparently you can use local size to speed up this computation even faster.
def lower_contigious (cmd: ContigiousNode):
    return f"{lower_karg(cmd.kres)} = {lower_karg(cmd.kargs[0])};"

def execute_contigious (context: ADCLContext, cmd: ContigiousNode):
    name = f"contigious_{context.get_prog_id()}"
    args, program_args = lower_args(cmd)

    # construct program
    program = context.get_program(name, f"""
__kernel void {name} (
    {program_args}
) {{
    const size_t _global_id = get_global_id(0);
    {lower_contigious(cmd)} 
}}           
    """.strip())
    
    # buffer args
    args = [context.get_buffer(buf_id) for buf_id in args]
   
    program(context.command_queue, (prod(cmd.shape), ), None, *args)  