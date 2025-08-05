from ....graph import *
from ....context import *
from ..karg import *
from ..cl_helper import *
from ..device import OpenCLDevice
from math import ceil

from .reduce import lower_reduce_op
from .binary import lower_binary
from .unary import lower_unary

def lower_reduce_elw_fuse (fused_cmd: ReduceElwFuse):
    res = ""
    for cmd in fused_cmd.nodes:
        match cmd:
            case BinaryNode():
                res += lower_binary(cmd)
            case UnaryNode():
                res += lower_unary(cmd)
            case AllocEntry():
                res += lower_temp_alloc(cmd)
            case ReduceNode():
                res += f"{lower_karg(cmd.kres)} = scratch[0];"
        res += "\n    "
    return res        

def init_reduce_elw_fuse (dev: OpenCLDevice, cmd: ReduceElwFuse):
    name = f"reduce_elw_fuse_{cmd.program_id}"
    args, program_args = lower_args(cmd)
    reduce_node = cmd.get_reduce()
    
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
    scratch[_y] = (_y < _l_size) ? {lower_karg(reduce_node.kargs[0])} : 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE); // waits until transfer to local memory is all finished

    // Reduction in local memory
    for (int _offset = _local_size / 2; _offset > 0; _offset >>= 1) {{
        if (_y < _offset) {{
            {lower_reduce_op(reduce_node.op, "scratch[_y]", "scratch[_y + _offset]")}
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}

    // Write result of this work-group to partial_sums
    if (_y == 0) {{
        int _global_id = _x; 
        {lower_reduce_elw_fuse(cmd)}
    }}
}}           
    """.strip()
    
    # get sizes
    sh = reduce_node.children_shapes[0]
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
