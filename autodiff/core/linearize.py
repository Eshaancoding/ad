from pprint import pprint
from ..context import Context
from ..node import Node
from toposort import toposort
from typing import Dict, List
from .helper import walk_graph
from .fusion import FuseBase, FuseType, ElwFuse, DPElwFuse, ReduceElwFuse, Procedure
from .fusion.helper import print_list
from .fusion.fuse_across import fuse_across
from .fusion.fuse_within import fuse_within 

def linearize (context: Context):
    for proc in context.procedure:
        g_dep = {}
        id_to_node = {}

        def test_toposort(n: Node, visited: Dict[int, int] = {}):
            n_id = n.id
            id_to_node[n_id] = n
            visited[n.id] = 1

            ids_dep = [child.id for child in n.children()]
            if n_id not in g_dep:
                g_dep[n_id] = list(set(ids_dep))
            else:
                g_dep[n_id].extend(ids_dep)
                g_dep[n_id] = list(set(g_dep[n_id]))
            
            for child in n.children():
                if not (child.id in visited):
                    test_toposort(child, visited)

            return n
        
        walk_graph(proc.nodes, test_toposort)
        
        toposort_res = list(toposort(g_dep))
        
        # Fuse!
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