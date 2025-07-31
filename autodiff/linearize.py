from .context import Context
from .node import Node
from toposort import toposort
from typing import Dict, List
from .helper import walk_graph, benchmark
from .fusion import *
from .graph import ConcatNode, ConstantNode
from time import time
from .context import Block, Proc

def linearize (proc: Block, already_decl: List[int] = []) -> Proc:
    g_dep = {}
    id_to_node = {}

    # Fill id to node
    def fill_id_to_node (n: Node, visited: Dict[int, int] = {}):
        # ConcatNode and ConstantNode is not folded by kernalize. Could be a todo for future releases
        # For now, just ignore them at linearize

        if (block_inner := n.get_block()) is not None:
            # TODO: double check whether already_decl works in intense circumstances
            # seems pretty... weird
            n.proc = linearize(block_inner, already_decl=list(id_to_node.keys()))
            id_to_node[n.id] = n
        elif not isinstance(n, ConcatNode) and not isinstance(n, ConstantNode): 
            id_to_node[n.id] = n

        return n

    walk_graph(proc.nodes, fill_id_to_node, walk_block=False)

    # Fill deps (g_dep)
    def fill_deps (n: Node, visited: Dict[int, int] = {}):
        n_id = n.id
        visited[n.id] = 1

        # any node with a block will *automatically* assume that it's related to all of its previous nodes defined
        # control opts are defined seperately (ex: taking expressions out of for)
        if n.get_block() is not None:
            g_dep[n_id] = list(g_dep.keys())
            return n

        ids_dep = n.kargs_child_ids()
        if n_id not in g_dep:
            g_dep[n_id] = ids_dep
        else:
            g_dep[n_id].extend(ids_dep)
            g_dep[n_id] = list(set(g_dep[n_id]))

        for child_id in ids_dep:
            if not child_id in visited:
                fill_deps(id_to_node[child_id], visited)

        return n

    for n in proc.nodes:
        fill_deps(n)
        
    # filter g dep
    for n_id in g_dep:
        g_dep[n_id] = list(filter(lambda item: item not in already_decl, g_dep[n_id]))
    g_dep = {key:val for key, val in g_dep.items() if len(val) > 0}
    
    # toposort
    toposort_res = list(toposort(g_dep))

    # Fuse!
    # TODO: watch for fusion node 80, 138
    fusion_ops: List[FuseBase] = [
        DPElwFuse,
        ReduceElwFuse,
        ElwFuse,
        Procedure # fusing procedure takes a while...
    ]
    
    def fn_fuse (toposort_res, id_to_node, op, fuse_type):
        # apply fuse operator until it can't
        ch = 1
        itr = 0
        while ch > 0:
            if fuse_type == FuseType.ACROSS_LAYER:
                toposort_res, ch = fuse_across(id_to_node, toposort_res, op)
            elif fuse_type == FuseType.WITHIN_LAYER:
                toposort_res, ch = fuse_within(id_to_node, toposort_res, op)

            if itr >= 1000: # iter stop
                print("FUSE OPERATION ITERATION PEAKED!!!") # alert the user; this shouldn't happen in most scenario
                break

        return toposort_res
            
    for op in fusion_ops:            
        if op().type == FuseType.ALL:
            st = time()
            toposort_res = fn_fuse(toposort_res, id_to_node, op, FuseType.WITHIN_LAYER) 
            toposort_res = fn_fuse(toposort_res, id_to_node, op, FuseType.ACROSS_LAYER) 
            end = time() 
            print(f"Fuse {op.__name__} took: ", end - st)
        else:
            toposort_res = benchmark(lambda: fn_fuse(toposort_res, id_to_node, op, op().type), name=f"Fuse Op: {op.__name__}")

    try:
        assert len(toposort_res) == 1, "Didn't fully capture procedure."
        assert len(toposort_res[0]) == 1, "Didn't fully capture procedure. "
        node = id_to_node[list(toposort_res[0])[0]]
        assert isinstance(node, FuseBase), "First node is not a FuseBase object"
    except Exception as e:
        # print the proc as of now
        for layer in toposort_res:
            print("\n================ Layer ================")
            for node in layer:
                print(id_to_node[node]) 
    
        raise e
    
    return Proc(node.nodes)
