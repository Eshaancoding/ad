from ..node import Node
from typing import Dict
from colored import stylize, fore

def indent(text: str, size: int = 1, prefix: str = "  ") -> str:
    return "\n".join((prefix*size) + line if line.strip() else line for line in text.splitlines())

# print one node or print multiple nodes

def format_node (n: Node, visited: Dict[int, int], level): 
    if level is not None:
        if level == 0:
            return "", visited
    
    s = str(n) + "\n"
    for child in n.children():
        p = ""
        if not (child.id in visited):
            p, visited = format_node(child, visited, level-1 if isinstance(level, int) else None)
            visited[child.id] = 1
        else:
            p = stylize(f"Intermediate {child.id}", fore("cyan"))
        s += indent(p)
        s += "\n"
    return s, visited

def format_graph (n, level):
    res = ""
    if isinstance(n, list) and len(n) > 0 and isinstance(n[0], Node):
        visited = {}
        for idx, node in enumerate(n):
            formatted_str, visited = format_node(node, visited, level)
            res += stylize(str(idx), fore("green")) + ": "
            res += formatted_str
    elif isinstance(n, Node):
        res, _ = format_node(n, {})
    else:
        raise TypeError(f"Invalid type {type(n)} in format graph")
    return res

def print_graph (n):
    print(format_graph(n, None))