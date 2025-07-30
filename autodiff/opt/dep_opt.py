from autodiff.context import Context
from autodiff.fusion.helper import get_res
from autodiff.node import Node

def dep_opt (context: Context):
    
    assert len(context.deps) > 0, "Empty dependency list"

    # iterate over the procedure and only keep the nodes in dep list
    def filter_node (node: Node): 
        r = get_res(node)
        if len(r) != 1:
            return node
        r = list(r)[0]
        
        if r not in context.deps:
            return None
        else:
            return node

    context.procedure[0].walk(filter_node)

    if len(context.procedure[0].nodes) == 0:
        raise Exception("Empty execution. Make sure to add resultant to dep list")
