from . import *

def lower_binary (cmd: BinaryNode):
    op_str = ""
    match cmd.op:
        case BinaryOp.ADD:
            op_str = "+"
        case BinaryOp.MULT:
            op_str = "+"

    return f"{lower_karg(cmd.kargs[0])} {op_str} {lower_args(cmd.kargs[1])}"

def execute_binary (context: ADCLContext, cmd: BinaryNode):
    name = f"binary_{context.get_prog_id()}"

    program = context.get_program(name, f"""
__kernel void {name} (
    {lower_args(cmd)}
) {{
    const size_t _global_id = get_global_id(0);
    
}}           
    """.strip())
    
    