from ....graph import *
from ....context import *
from ..context import ADCLContext
from ..karg import *
import numpy as np

from .binary import lower_binary
from .unary import lower_unary

def lower_dp_elw_fuse (fused_cmd: DPElwFuse):
    res = ""
    for cmd in fused_cmd.nodes:
        match cmd:
            case BinaryNode():
                res += lower_binary(cmd)
            case UnaryNode():
                res += lower_unary(cmd)
            case AllocEntry():
                res += lower_temp_alloc(cmd)
            case DotProdNode():
                res += f"{lower_karg(cmd.kres)} = _value;"
        res += "\n    "
    return res        

def execute_dp_elw_fuse (context: ADCLContext, cmd: DPElwFuse):
    dpnode = cmd.get_dp()
    name = f"dp_elw_fuse_{context.get_prog_id()}"
    args, program_args = lower_args(cmd)

    # construct program
    program = context.get_program(name, f"""
__kernel void {name} (
    {program_args},
    int _wA,
    int _wB
) {{
    int _tx = get_global_id(0); // Column index in C; --> output size
    int _ty = get_global_id(1); // Row index in C;    --> batch size

    float _value = 0.0f;
    for (int _k = 0; _k < _wA; ++_k) {{
        int _x = _ty;
        int _y = _k;
        float _elementA = {lower_karg(dpnode.kargs[0])};

        _x = _k;
        _y = _tx;
        float _elementB = {lower_karg(dpnode.kargs[1])};
        _value += _elementA * _elementB;
    }}

    int _x = _ty;
    int _y = _tx;
    int _global_id = _ty * _wB + _tx;
    {lower_dp_elw_fuse(cmd)}
}}           
    """.strip())
    
    # buffer args
    args = [context.get_buffer(buf_id) for buf_id in args]
    args.append(np.int32(dpnode.children_shapes[0][1])) # _wA = input size
    args.append(np.int32(dpnode.children_shapes[1][1])) # _wB = output size
   
    program(
        context.command_queue,                  # Command queue
        (dpnode.shape[1], dpnode.shape[0]),     # global size 
        None,                                   # local size
        *args                                   # arguments
    )