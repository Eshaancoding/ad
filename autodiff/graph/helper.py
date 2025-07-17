from copy import deepcopy
from typing import Callable, Tuple, List
from ..node import Node
from ..expr import *

# shape helper
def calc_stride(shape: List[int]) -> List[Expression]:
    n = len(shape)
    strides: List[Expression] = [Val(Constant(1)) for _ in range(n)]
    for i in reversed(range(n - 1)):
        next_val = strides[i + 1].get_const() * shape[i + 1]
        strides[i] = Val(Constant(next_val))
    return strides

def global_to_ndim(index: Expression, shape: List[int]) -> List[Expression]:
    strides = calc_stride(shape)
    return [
        Remainder(
            Div(index, strides[i]),
            Val(Constant(shape[i]))
        )
        for i in range(len(shape))
    ]

def ndim_to_global(dim: List[Expression], shape: List[int]) -> Expression:
    assert len(dim) == len(shape), f"Dimension and the shape length mismatch\ndim: {dim}, shape: {shape}"

    strides = calc_stride(shape)
    global_expr = Mult(dim[0], strides[0])
    for i in range(1, len(shape)):
        global_expr = Add(
            global_expr,
            Mult(
                dim[i], 
                strides[i]
            )
        )
    return global_expr