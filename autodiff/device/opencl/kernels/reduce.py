from ....graph import *
from ....context import *
from ..karg import *
from math import ceil
from ..device import OpenCLDevice
from ..cl_helper import *

def lower_reduce_op (op: ReduceOp, orig:str, new:str):
    match op:
        case ReduceOp.SUM:
            return f"{orig} += {new};"
        case ReduceOp.MAX:
            return f"{orig} = max({orig}, {new});"

def init_reduce (dev: OpenCLDevice, cmd: ReduceNode):
    name = f"reduce_{cmd.program_id}"
    args, program_args = lower_args(cmd)
    
    # construct program
    program_str = f"""
__kernel void {name} (
    {program_args},
    __local float* scratch,
    int _l_size
) {{
    
    // global work size = local work size * number of groups
    int _x = get_group_id(0);
    int _y = get_local_id(0);
    int _local_size = get_local_size(0);

    // Load data into local memory
    scratch[_y] = (_y < _l_size) ? {lower_karg(cmd.kargs[0])} : 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE); // waits until transfer to local memory is all finished

    // Reduction in local memory
    for (int _offset = _local_size / 2; _offset > 0; _offset >>= 1) {{
        if (_y < _offset) {{
            {lower_reduce_op(cmd.op, "scratch[_y]", "scratch[_y + _offset]")}
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}

    // Write result of this work-group to partial_sums
    if (_y == 0) {{
        {lower_karg(cmd.kres)} = scratch[0];
    }}
}}           
    """.strip()
    
    # get sizes
    sh = cmd.children_shapes[0]
    vec_size = sh[0] 
    reduce_size = sh[1]
    local_size = ceil(reduce_size / 2) * 2
   
    # buffer args
    args = [Buffer(dev.buffers[buf_id]) for buf_id in args]
    args.append(LocalMem(local_size))
    args.append(Int(reduce_size))

    global_size = (int(local_size * vec_size), )
    local_size = (int(local_size), )
    
    return build_kernel(dev, name, program_str, args, global_size, local_size)
