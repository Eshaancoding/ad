from autodiff.context import Proc
from autodiff.fusion.helper import get_deps
from autodiff.graph.compute.binary import BinaryNode
from autodiff.node import Node


def set_in_place (proc: Proc):
    repl = {}

    # reset cache for get deps 
    get_deps.cache_clear()

    def binary_set (node: Node, _):
        if isinstance(node, BinaryNode) and node.in_place:
            node.kres.rename(node.id, node.left.id)
            repl[node.id] = node.left.id
            node.id = node.left.id
        return node

    proc.walk(binary_set)

    def deps_set (node: Node, _):
        for fr in repl:
            to = repl[fr]
            for idx in range(len(node.kargs)):
                node.kargs[idx].rename(fr, to)
        return node

    proc.walk(deps_set)

    return proc
