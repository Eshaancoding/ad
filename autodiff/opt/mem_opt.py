"""
for cmd 
    1. get result
    2. get deps
    3. for dep in deps:
        1. if there's no future deps after the cmd, then you can replace result with dep 
        2. replace_var = (result, dep) 
"""

from autodiff.context import Proc, context
from autodiff.fusion.helper import get_deps, get_res
from autodiff.graph.compute.binary import BinaryNode, BinaryOp
from autodiff.graph.compute.dotprod import DotProdNode
from autodiff.graph.compute.reduce import ReduceNode
from autodiff.graph.data.contigious import ContigiousNode
from autodiff.node import Node
from typing import Dict, List, Set, Tuple

idx = 0

def mem_opt (proc: Proc):
    global idx
    deps = context.deps + list(context.dep_replace.values())

    # track the dep_list
    ref_location: Dict[int, List[Tuple[int, int]]] = {}
    res_location: Dict[int, int] = {}
    in_place_ids: Set[int] = set()

    def track_res_loc (node: Node, proc_id: int):
        global idx

        for r in get_res(node):
            if r not in deps:
                ref_location[r] = []
                res_location[r] = proc_id

        for d in get_deps(node):
            if d in ref_location:
                ref_location[d].append((idx, proc_id))

        if isinstance(node, BinaryNode) and node.in_place:
            in_place_ids.add(node.id)
    
        idx += 1

        return node

    idx = 0  # reset
    proc.walk(track_res_loc)

    # determine if it is suitable to replace --> append
    to_replace = {}
    def get_replace_itr (node: Node, proc_id: int):
        global idx

        # ensure node type
        if not isinstance(node, Node):
            idx += 1
            return node # skip if not Node

        # Get result
        res = list(get_res(node))
        if len(res) != 1:
            idx += 1
            return node # skip if no res
        res = res[0]

        # don't replace with reduce operations, dot prod, or contigious
        # furthermore, don't replace if node is in deps in the first place
        if isinstance(node, DotProdNode) or \
            isinstance(node, ReduceNode) or \
            isinstance(node, ContigiousNode) or \
            res in in_place_ids or \
            res in deps:

            idx += 1
            return node
        
        repl = [] 
        for dep in get_deps(node):
            if dep not in res_location: 
                continue

            # ensure that the proc id for dep is the same as the proc id
            if res_location[dep] != proc_id:
                continue
            
            # if there's no future deps of dep after cmd current, then you can replace
            v = list(filter(lambda x: x[0] > idx or x[1] != proc_id, ref_location[dep]))
            if len(v) > 0:
                continue

            repl.append((res, dep)) 
            
        # append to replace
        if len(repl) > 0:
            repl = repl[0] # only choose the first swap memory
            to_replace[repl[0]] = repl[1]
             
         
        # add idx and return
        idx += 1
        return node

    idx = 0
    proc.walk(get_replace_itr)

    # resolve conflicts
    to_replace_items = list(to_replace.items())
    for idx, (res, ref) in enumerate(to_replace_items):
        for idx_future in range(idx+1, len(to_replace_items)):
            key, value = to_replace_items[idx_future] 
            if value == res:
                to_replace_items[idx_future] = (
                    key,
                    ref
                )

    to_replace = {k:v for k,v in to_replace_items}

    # actually replac
    def replace (node: Node, _):
        # if we have the  
        for k, v in to_replace.items():
            node.rename(k,v)

        return node

    proc.walk(replace)

    return proc
