from pprint import pprint
from typing import Any, Dict, List, Set
from autodiff.context import Context
from autodiff.graph.compute.binary import BinaryNode
from autodiff.graph.compute.unary import UnaryNode
from autodiff.graph.data.constant import ConstantNode
from autodiff.graph.tensor import Tensor
from autodiff.node import Node
from autodiff.helper import walk_graph
from autodiff.print_graph import pg

def _intern_repeat_opt (context: Context):
    # first str is the type, 2nd is the list of kargs concat by string
    pot_nodes: Dict[Any, List[Node]] = {}
    
    def track_nodes (node: Node, _):
        # don't add any for loops 

        if node.get_block() is not None: 
            return node

        # don't do tensor or constant nodes
        if isinstance(node, Tensor) or isinstance(node, ConstantNode):
            return node

        hs = tuple([
            ch.repeat_helper(is_child=True)
            for ch in node.children()
        ]) + tuple(node.repeat_helper(is_child=False))

        if hs in pot_nodes:
            pot_nodes[hs].append(node)
        else:
            pot_nodes[hs] = [node] 

        return node

    context.procedure[0] = walk_graph(
        context.procedure[0],
        track_nodes
    )

    pot_nodes = [
        nodes
        for nodes in pot_nodes.values()
        if len(nodes) > 1
    ]

    # given matches, replace
    def replace_node (node: Node, _):
        # check for matches
        for nodes in pot_nodes:
            to_replace = nodes[0]
            to_search = set([v.id for v in nodes[1:]])
            if node.id in to_search:
                return to_replace
        return node

    context.procedure[0] = walk_graph(context.procedure[0], replace_node)

    return len(pot_nodes)

# A somewhat brute force approach to this problem, but oh well...
def repeat_opt (context: Context):
    d = 1
    itr = 0
    while d > 0:
        d = _intern_repeat_opt(context)
        itr += 1
        if itr >= 100:
            print("Repeat opt max iteration passed") 
            break
