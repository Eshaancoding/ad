from ....graph import *
from ....context import *
from ..karg import *
from math import prod
from ..cl_helper import *
from ..device import OpenCLDevice

def lower_binary (cmd: BinaryNode):
    op_str = ""
    match cmd.op:
        case BinaryOp.ADD:
            op_str = "+"
        case BinaryOp.MULT:
            op_str = "*"

    return f"{lower_karg(cmd.kres)} = {lower_karg(cmd.kargs[0])} {op_str} {lower_karg(cmd.kargs[1])};"

def init_binary (dev:OpenCLDevice, cmd: BinaryNode) -> Callable:
    name = f"binary_{cmd.program_id}"
    args, program_args = lower_args(cmd)
    program_str = f"""
__kernel void {name} (
    {program_args}
) {{
    const size_t _global_id = get_global_id(0);
    {lower_binary(cmd)} 
}}           
    """.strip()
    
    # buffer args
    args = [Buffer(dev.buffers[buf_id]) for buf_id in args]
   
    return build_kernel(dev, name, program_str, args, (prod(cmd.shape), ), None)
