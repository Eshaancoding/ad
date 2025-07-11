from copy import deepcopy
from typing import Callable, Tuple, List
from ..node import Node

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
                if isinstance(r, Node):
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

    # # Check for pattern match first,
    # for pattern, replacement in pattern_dict:
    #     if node.type_eq(pattern): # only match the type
    #         r = replacement(node)
    #         if isinstance(r, Node):
    #             return r
    
    # # Otherwise, recurse into children
    # new_children = [replace_patterns(child, pattern_dict) for child in node.children]
    # new_node = deepcopy(node)
    # new_node.children = new_children

    # return new_node
