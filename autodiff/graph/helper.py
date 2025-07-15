from copy import deepcopy
from typing import Callable, Tuple, List
from ..node import Node
from ..expr import *

# Good for debugging
def indent(text: str, size: int = 1, prefix: str = "  ") -> str:
    return "\n".join((prefix*size) + line if line.strip() else line for line in text.splitlines())

# Good for pattern matching, ir opts, etc.
def _internal_repl_pat (node: Node, pattern_dict: List[Tuple[Node, Callable[[Node], Node]]]) -> Tuple[Node, bool]:
    idx = 0
    while idx < 100:  # max reruns of the graph
        # Check for pattern match first,
        for pattern, replacement in pattern_dict:
            if node.type_eq(pattern): # only match the type
                r = replacement(node)
                if r is not None and isinstance(r, Node):
                    return (r, True)
        
        # Otherwise, recurse into children
        new_children = []
        did_r = False
        for child in node.children:
            new_child, did_replace = _internal_repl_pat(child, pattern_dict)
            new_children.append(new_child)
            did_r = did_r or did_replace

        node = deepcopy(node)
        node.children = new_children

        # if we did not replace, then return this node (and False, not changed)
        if not did_r: 
            return (node, False)
        
        # however, if we did replace something on the computation graph, then rerun the replacement operation
        idx += 1
        
    # ideally, shouldn't go into this area of code
    print("=================== REACHED PASSED MAX PATTERN ITERATION ============")
    return (node, False)

def replace_patterns(node: Node, pattern_dict: List[Tuple[Node, Callable[[Node], Node]]]) -> Node:
    v, _ = _internal_repl_pat(node, pattern_dict)    
    return v

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