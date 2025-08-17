from pprint import pprint
from typing import Dict
from autodiff.context import Context
from autodiff.graph.data.feeder import Feeder
from autodiff.graph.data.receiver import Receiver
from autodiff.node import Node
from autodiff.helper import walk_graph

def _intern_repeat_opt (context: Context):
    pot_nodes: Dict[int, Node] = {}
    
    def track_nodes (node: Node, _visited):
        # don't add any for loops or control
        if node.get_block() is not None: 
            return node

        # don't add any receiver or feeder nodes
        if isinstance(node, Feeder) or isinstance(node, Receiver):
            return node

        # if nodes with a single child, add to pot nodes
        if len(node.children()) == 1:
            pot_nodes[node.id] = node
            return node

        # for nodes with multiple children, check if any of those children has no children
        could_add = False
        for n in node.children():
            could_add = could_add or len(n.children()) == 0
        if could_add:
            pot_nodes[node.id] = node

        return node

    context.procedure[0] = walk_graph(
        context.procedure[0],
        track_nodes
    )

    # find matches (O(n^2), could be improved)
    items_potnodes = pot_nodes.items()
    matches = {}
    for idx_one, (node_one_id, node_one) in enumerate(items_potnodes):
        for idx_two, (node_two_id, node_two) in enumerate(items_potnodes):
            if idx_one == idx_two:
                continue
            
            if node_one.node_eq(node_two):
                if node_two_id not in matches:
                    matches[node_one_id] = node_two_id
    
    # given matches, replace
    def replace_node (node: Node, _):
        # check for matches
        for to_replace, to_search in matches.items():
            if node.id == to_search:
                return pot_nodes[to_replace]
        return node

    context.procedure[0] = walk_graph(context.procedure[0], replace_node)

    return len(matches)

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
