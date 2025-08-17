from autodiff.node import Node
from typing import Set, List, Dict
from toposort import toposort
from ..helper import walk_graph

def get_deps (node: Node, step_proc:bool=False, replace:Dict={}) -> Set[int]:
    """ Note, uses kernalize result """
    from .base import FuseBase
    if isinstance(node, Node):
        if step_proc and (proc := node.get_proc()) is not None:
            res_defined = get_res(node, True)

            ret = set()
            for n in proc.procedure:
                ret.update([
                    (replace[d] if d in replace else d)
                    for d in get_deps(n, True) if d not in res_defined
                ])
            ret = set(filter(lambda x: x is not None, ret))

            return ret 
        else:
            return set({
                (replace[child_id] if child_id in replace else child_id)
                for child_id in node.kargs_child_ids()
            })
    elif isinstance(node, FuseBase):
        res_defined = get_res(node, step_proc)

        ret = set()

        """
        for n in node.nodes:
            ret.update([
                (replace[d] if d in replace else d)
                for d in get_deps(n, True) if d not in results
            ]) 
        """
        for n in node.nodes:
            for d in get_deps(n, True):
                if d not in res_defined:
                    v = replace[d] if d in replace else d
                    ret.add(v)
        ret = set(filter(lambda x: x is not None, ret))

        return ret
    else:
        raise TypeError(f"Invalid type in calc_deps: {type(node)}")

def print_toposort (toposort_res, id_to_node):
    for idx, layer in enumerate(toposort_res):
        print(f"\n================ Layer #{idx+1} ================")
        for node in layer:
            print(id_to_node[node]) 
    
def get_res (node: Node, step_proc:bool=False) -> Set[int]:
    from .base import FuseBase
    if isinstance(node, Node):
        if step_proc and (proc := node.get_proc()) is not None:
            ret = []
            for n in proc.procedure:
                ret.extend(get_res(n, True))
            return set(ret) 
        else:
            return set([node.id]) # same as the kres id
    elif isinstance(node, FuseBase):
        r = []
        for n in node.nodes:
            r.extend(get_res(n))
        return set(r)
    else:
        raise TypeError(f"Invalid type in calc_res: {type(node)}")
