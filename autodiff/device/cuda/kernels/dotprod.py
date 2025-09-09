from ....graph import *
from ....context import *
from ..karg import *
from ..device import CudaDevice
from ..cuda_helper import *

def init_dotprod (dev: CudaDevice, cmd: DotProdNode):
    name = f"dotprod_{cmd.program_id}"
    args, program_args = lower_args(cmd)

    # construct program
    program_str = f"""
__kernel void {name} (
    {program_args},
    int _wA
) {{
    int _global_id = blockIdx.x;
    int _tx = blockIdx.x;
    int _ty = blockIdx.y;

    float _value = 0.0f;
    for (int _k = 0; _k < _wA; ++_k) {{
        int _x = _ty;
        int _y = _k;
        float _elementA = {lower_karg(cmd.kargs[0])};

        _x = _k;
        _y = _tx;
        float _elementB = {lower_karg(cmd.kargs[1])};
        _value += _elementA * _elementB;
    }}

    int _x = _ty;
    int _y = _tx;
    {lower_karg(cmd.kres)} = _value;   
}}           
    """.strip()
    
    # buffer args
    args = [Buffer(dev.buffers[buf_id]) for buf_id in args]
    args.append(Int(cmd.children_shapes[0][1]))

    return build_kernel(
        name,
        program_str,
        args,
        dev.arch,
        (cmd.shape[1], cmd.shape[0], 1),
    )
