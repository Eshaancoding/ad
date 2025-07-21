from copy import deepcopy
from .node import Node
from .graph import ConstantNode

def opt_node (node: Node) -> Node:
    ##################################################
    # Constant + data permutation folding

    """
    def replace_cdp_one (n:Node):
        target_shape = n.shape
        child = n.c(0)
        child.dim = target_shape
        return child
    
    cdp_one = [
        (ConstantNode(0.0, [1, 2]).permute([1, 0]), replace_cdp_one),
        (ConstantNode(0.0, [1, 2]).view([2]), replace_cdp_one),
        (ConstantNode(0.0, [1, 2])[0], replace_cdp_one),
        (ConstantNode(0.0, [1, 2]).broadcast(0, 2), replace_cdp_one),
    ]
    """
    
    ##################################################
    # Constant + sum folding

    # return
    # return replace_patterns(node, [
    #     *cdp_one 
    # ]) 
    
    return node
    
    
############ REDO OPT NODE LIKE EXPR/SIMPLIFY
### Probably need all the features in expr