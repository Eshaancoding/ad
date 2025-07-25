from ....graph import *
from ....context import *
from ..context import ADCLContext
from ..karg import *
from math import ceil
import numpy as np
import pyopencl as cl

def lower_reduce_op (op: ReduceOp, orig:str, new:str):
    match op:
        case ReduceOp.SUM:
            return f"{orig} += {new};"
        case ReduceOp.MAX:
            return f"{orig} = max({orig}, {new});"

def execute_reduce (context: ADCLContext, cmd: ReduceNode):
    name = f"reduce_{context.get_prog_id()}"
    args, program_args = lower_args(cmd)
    
    # construct program
    program = context.get_program(name, f"""
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
    """.strip())
    
    # get sizes
    sh = cmd.children_shapes[0]
    vec_size = sh[0] 
    reduce_size = sh[1]
    local_size = ceil(reduce_size / 2) * 2
   
    # buffer args
    args = [context.get_buffer(buf_id) for buf_id in args]
    args.append(cl.LocalMemory(local_size * np.dtype(np.float32).itemsize))
    args.append(np.int32(reduce_size))
  
    program(
        context.command_queue,        # Command queue
        (local_size * vec_size, ),     # global size 
        (local_size, ),               # local size
        *args                         # arguments
    )