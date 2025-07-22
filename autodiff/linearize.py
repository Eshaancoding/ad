from .context import Context
from .node import Node
from toposort import toposort
from typing import Dict, List
from .helper import walk_graph
from .fusion import *
from .graph import ConcatNode, ConstantNode

def linearize (context: Context):
    for proc in context.procedure:
        g_dep = {}
        id_to_node = {}

        # Fill id to node
        def fill_id_to_node (n: Node, visited: Dict[int, int] = {}):
            # ConcatNode and ConstantNode is not folded by linearize. Could be a todo for future releases
            if not isinstance(n, ConcatNode) and not isinstance(n, ConstantNode): 
                id_to_node[n.id] = n

        walk_graph(proc.nodes, fill_id_to_node)

        # Fill deps (g_dep)
        def fill_deps (n: Node, visited: Dict[int, int] = {}):
            n_id = n.id
            visited[n.id] = 1

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
        
        toposort_res = list(toposort(g_dep))
        
        # Fuse!
        # TODO: watch for fusion node 80, 138
        fusion_ops: List[FuseBase] = [
            DPElwFuse,
            ReduceElwFuse,
            ElwFuse,
            Procedure
        ]
        
        def fn_fuse (toposort_res, id_to_node, context, op, fuse_type):
            # apply fuse operator until it can't
            ch = 1
            itr = 0
            while ch > 0:
                if fuse_type == FuseType.ACROSS_LAYER:
                    toposort_res, ch = fuse_across(context, id_to_node, toposort_res, op)
                elif fuse_type == FuseType.WITHIN_LAYER:
                    toposort_res, ch = fuse_within(context, id_to_node, toposort_res, op)

                if itr >= 1000: # iter stop
                    print("FUSE OPERATION ITERATION PEAKED!!!") # alert the user; this shouldn't happen in most scenario
                    break

            return toposort_res
                
        for op in fusion_ops:            
            if op().type == FuseType.ALL:
                toposort_res = fn_fuse(toposort_res, id_to_node, context, op, FuseType.WITHIN_LAYER) 
                toposort_res = fn_fuse(toposort_res, id_to_node, context, op, FuseType.ACROSS_LAYER) 
            else:
                toposort_res = fn_fuse(toposort_res, id_to_node, context, op, op().type) 

        try:
            proc = list(id_to_node.values())[0].nodes
            for n in proc:
                print(n)
        except:
            print_list(id_to_node, toposort_res)
            raise Exception("Can't ret proc")