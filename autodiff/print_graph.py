from .node import Node
from typing import Dict
from colored import stylize, fore

def indent(text: str, size: int = 1, prefix: str = r"  ") -> str:
    return "\n".join((prefix*size) + line if line.strip() else line for line in text.splitlines())

def combine_dict (one:Dict, two:Dict):
    for key in two:
        if key not in one:
            one[key] = two[key]
    return one

# print one node or print multiple nodes
def format_node (n: Node, visited: Dict[int, int], level): 
    visited[n.id] = 1
    if level is not None:
        if level == 0:
            return "", visited

    s = str(n).split("-->")[0] + "\n"
    if (block := n.get_block()) is not None:
        vis_two, st = format_graph(block.nodes, level-1 if isinstance(level, int) else None)
        s += indent(st) + "\n"
        visited = combine_dict(visited, vis_two)
    
    # NOTE: Will display the Concat node, even though it's folded at kernalize
    for child in n.children():
        p = ""
        if child.id not in visited:
            p, visited = format_node(child, visited, level-1 if isinstance(level, int) else None)
            visited[child.id] = 1
        else:
            p = stylize(f"{child.id} = Intermediate", fore("cyan"))
        s += indent(p)
        s += "\n"

    return s, visited

def format_graph(n, level):
    res = ""
    visited = {}
    if isinstance(n, list) and len(n) > 0 and isinstance(n[0], Node):
        for idx, node in enumerate(n):
            formatted_str, visited = format_node(node, visited, level)
            res += stylize(str(idx), fore("green")) + ": "
            res += formatted_str
        return visited, res
    elif isinstance(n, list) and len(n) == 0:
        return {}, "[]"
    elif isinstance(n, Node):
        return format_graph([n], level)
    else:
        raise TypeError(f"Invalid type {type(n)} in format graph")


def pg (n):
    """
    Note that print graph will show concat node, even though it's folded
    at kernalize.
    """
    _, st  = format_graph(n, None)
    print(st)
