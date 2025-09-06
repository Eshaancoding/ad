from ....graph import *
from ....context import *
from ....fusion import *
from ....alloc import *
from ..karg import *
from ..device import OpenCLDevice
from ..cl_helper import *
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
                assert cmd.is_temp, "not temp alloc entry in elwfuse..."
                res += lower_temp_alloc(cmd)
        res += "\n    "
    return res        
         
def init_elwfuse (dev: OpenCLDevice, cmd: ElwFuse):
    name = f"elwfuse_{cmd.program_id}"
    args, program_args = lower_args(cmd)
    
    # construct program
    program_str = f"""
__kernel void {name} (
    {program_args}
) {{
    const size_t _global_id = get_global_id(0);
    {lower_elwfuse(cmd)} 
}}           
    """.strip()
    
    # buffer args
    args = [Buffer(dev.buffers[buf_id]) for buf_id in args]

    return build_kernel(dev, name, program_str, args, (prod(cmd.init_node.shape), ), None)
