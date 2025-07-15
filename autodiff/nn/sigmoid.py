from ..node import Node

def sigmoid (n: Node):
    return 1 / (1 + (-n).exp())