from typing import Dict
from autodiff.context import Context
from autodiff.node import Node
from autodiff.helper import walk_graph

def _intern_repeat_opt (context: Context):
    
    pot_nodes: Dict[int, Node] = {}
    deps = context.deps
    
    def track_nodes (node: Node, _visited):
        # check if dep two node
        is_dep_two = len(node.children()) > 0
        for n in node.children():
            is_dep_two = is_dep_two and len(n.children()) == 0

        if is_dep_two:
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
                if node_two_id not in matches and node_two_id not in deps:
                    matches[node_one_id] = node_two_id
    
    # given matches, replace
    def replace_node (node: Node, _):
        # check for matches
        for to_replace, to_search in matches.items():
            if node.id == to_search:
                return pot_nodes[to_replace]
        return node

    #TODO: Fix where the node to be replaced is a parent node
    """
    Test case:
    x = Tensor.randn([8,4]) 
    y = x.exp2().log2()
    z = x.exp2() + 3

    y.keep()
    z.keep()

    execute()
    """

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
