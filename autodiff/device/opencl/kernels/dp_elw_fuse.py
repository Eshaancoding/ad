from ....graph import *
from ....context import *
from ..karg import *
from ..device import OpenCLDevice
from ..cl_helper import *
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

def init_dp_elw_fuse (dev: OpenCLDevice, cmd: DPElwFuse):
    dpnode = cmd.get_dp()
    name = f"dp_elw_fuse_{cmd.program_id}"
    args, program_args = lower_args(cmd)

    # construct program
    program_str = f"""
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
    """.strip()
    
    # buffer args
    args = [Buffer(dev.buffers[buf_id]) for buf_id in args]
    args.append(Int(dpnode.children_shapes[0][1]))
    args.append(Int(dpnode.children_shapes[1][1]))
   
    return build_kernel(dev, name, program_str, args, (dpnode.shape[1], dpnode.shape[0]), None)
