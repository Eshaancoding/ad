from autodiff.context import Context
from autodiff.fusion.helper import get_res
from autodiff.graph.compute.binary import BinaryNode
from autodiff.graph.data.receiver import Receiver
from autodiff.graph.tensor import Tensor
from autodiff.node import Node
from autodiff.helper import walk_graph

def dep_opt (context: Context):
    
    # iterate over the procedure and only keep the nodes in dep list
    # note using walk graph 
    def filter_node (node: Node, _): 
        # keep control nodes
        if not isinstance(node, Node) or node.get_block() is not None:
            return node

        # leave receiver nodes alone
        if isinstance(node, Receiver): 
            return node
    
        # leave binary ops (+=, -=, etc.) alone
        if isinstance(node, BinaryNode) and node.in_place: 
            return node

        # leave tensor declarations alone 
        if isinstance(node, Tensor):
            return node

        return None

    context.procedure[0] = walk_graph(
        context.procedure[0], 
        filter_node, 
        walk_block=True, 
        walk_child=False
    )

    if len(context.procedure[0].nodes) == 0:
        raise Exception("Empty execution. Make sure to add resultant to dep list")

    # if there's any block nodes that has length of 0, then return None
    # we are kind of using our own logic for walk graph - although could be replaced easily
    def filter_len_none (node: Node):
        if (r := node.get_block()) is not None:
            if len(r.nodes) == 0:
                return None 
            else:
                for idx, n in enumerate(r.nodes):
                    r.nodes[idx] = filter_len_none(n)
                r.nodes = list(filter(lambda x: x is not None, r.nodes))
    
        return node

    for idx, node in enumerate(context.procedure[0].nodes):
        context.procedure[0].nodes[idx] = filter_len_none(node)
    context.procedure[0].nodes = list(filter(lambda x: x is not None, context.procedure[0].nodes))
    
