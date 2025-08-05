from ..graph import Node
from ..context import Context
from . import FuseBase
from .helper import clean_toposort_res, print_toposort, resolve_one_to_many, resolve_many_to_one, resolve_circular_dep
from typing import Tuple, Dict
from pprint import pprint

def fuse_across (id_to_node, toposort_res, fuse_op: FuseBase) -> Tuple[Dict, int]:
    from ..context import context
    ch = 0
    for i in range(len(toposort_res)-1):
        layer_one = toposort_res[i] 
        layer_two = toposort_res[i+1]
         
        # iterate over first node
        matches = {}
        for id_one in layer_one:
            node_one: Node = id_to_node[id_one]
           
            # iterate over second nodes
            for id_two in layer_two:
                node_two: Node = id_to_node[id_two]
                
                # add to potential matches
                could_fuse = fuse_op.could_fuse(node_one, node_two)
                if could_fuse:
                    if id_one in matches:
                        matches[id_one].append(id_two)
                    else:
                        matches[id_one] = [id_two]
       
        # ---- resolve conflicts in matches ----
        matches = resolve_one_to_many(matches, id_to_node, True)
        matches = resolve_many_to_one(matches, id_to_node, True) 
        matches = resolve_circular_dep(matches)
        
        # ---- replace matches ---- 
        for id_one in matches:
            id_two = matches[id_one]
            
            node_one = id_to_node[id_one]
            node_two = id_to_node[id_two]
            
            # create new node
            new_node = fuse_op()
            new_node.add(node_one)
            new_node.add(node_two)
            
            # update id_to_node
            id_to_node[new_node.fuse_id] = new_node
            del id_to_node[id_one]
            del id_to_node[id_two]
            
            # update toposort res
            toposort_res[i] = set(filter(lambda x: x != id_one, toposort_res[i]))
            toposort_res[i+1] = set(filter(lambda x: x != id_two, toposort_res[i+1]))
            toposort_res[i].add(new_node.fuse_id)
    
        ch += len(matches)
        
    # iterate over toposort, and filter sets that are 0 
    toposort_res = clean_toposort_res(toposort_res, id_to_node)
    return toposort_res, ch
