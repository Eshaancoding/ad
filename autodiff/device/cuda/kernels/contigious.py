from ....graph import *
from ....context import *
from ..karg import *
from math import prod
from ..device import CudaDevice
from ..cuda_helper import *

# Apparently you can use local size to speed up this computation even faster.
def lower_contigious (cmd: ContigiousNode):
    return f"{lower_karg(cmd.kres)} = {lower_karg(cmd.kargs[0])};"

def init_contigious (dev: CudaDevice, cmd: ContigiousNode):
    name = f"contigious_{cmd.program_id}"
    args, program_args = lower_args(cmd)

    # construct program
    program_str = f"""
extern "C" __global__ void {name} (
    {program_args}
) {{
    int _global_id = blockIdx.x;
    {lower_contigious(cmd)} 
}}           
    """.strip()
    
    # buffer args
    args = [Buffer(dev.buffers[buf_id]) for buf_id in args]
   
    return build_kernel(
        name, 
        program_str, 
        args, 
        (prod(cmd.shape), 1, 1)
    ) 
