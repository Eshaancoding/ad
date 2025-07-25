from ....graph import *
from ....context import *
from ..context import ADCLContext
from ..karg import *
import numpy as np

def execute_dotprod (context: ADCLContext, cmd: DotProdNode):
    name = f"dotprod_{context.get_prog_id()}"
    args, program_args = lower_args(cmd)

    # construct program
    program = context.get_program(name, f"""
__kernel void {name} (
    {program_args},
    int _wA
) {{
    const size_t _global_id = get_global_id(0);
    int _tx = get_global_id(0); // Column index in C; --> output size
    int _ty = get_global_id(1); // Row index in C;    --> batch size

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
    """.strip())
    
    # buffer args
    args = [context.get_buffer(buf_id) for buf_id in args]
    args.append(np.int32(cmd.children_shapes[0][1])) # _wA = input size
   
    program(
        context.command_queue,   # Command queue
        (cmd.shape[1], cmd.shape[0]),     # global size 
        None,                    # local size
        *args                    # arguments
    )