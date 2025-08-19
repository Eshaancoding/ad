from copy import deepcopy
from numpy._core.multiarray import set_datetimeparse_function
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

def find_all_circular (dependencies):
    all_paths = []
    def dfs(node, path):
        if node in path:
            idx = path.index(node)
            cycle = path[idx:] + [node]
            all_paths.append(cycle)
            return
        path.append(node)
        for dep in dependencies.get(node, []):
            dfs(dep, path.copy())
        path.pop()

        return None

    for start in dependencies:
        dfs(start, [])

    # Optionally: filter duplicate cycles (same nodes in different order)
    unique_cycles = []
    seen_sets = set()
    for cycle in all_paths:
        cycle_set = frozenset(cycle)
        if cycle_set not in seen_sets:
            seen_sets.add(cycle_set)
            unique_cycles.append(cycle)
    return unique_cycles

def find_a_circular_dep (dependencies):
    def dfs(node, path):
        if node in path:
            idx = path.index(node)
            cycle = path[idx:]
            return cycle
        path.append(node)
        for dep in dependencies.get(node, []):
            v = dfs(dep, path.copy())
            if v is not None:
                return v
        path.pop()

        return None

    for start in dependencies:
        v = dfs(start, [])
        if v is not None:
            return v

    return None

def resolve_circular_deps (g_dep, id_to_node):
    from .base import FuseBase
    is_fuse = lambda x: isinstance(x, FuseBase)

    # this function basically attempts to remove the two --> one dependency line
    while (path := find_a_circular_dep(g_dep)) is not None:
        from pprint import pprint

        pprint(find_all_circular(g_dep))
        print()
        print()

        dep_one, dep_two = path[0], path[-1]
        node_one = id_to_node[dep_one]
        node_two = id_to_node[dep_two]

        assert is_fuse(node_two), \
            f"Unsupported resolve dependencies type {type(node_two)}\n{node_one}\n{node_two}\n{path}"

        # get results of each node
        results_one = get_res(node_one)

        # go through each nodes and dependencies for node_one and node_two
        deps_two = set({
            idx
            for idx, node in enumerate(node_two.nodes)
            for d in get_deps(node) if d in results_one
        }) if is_fuse(node_two) else set() 

        if len(deps_two) != 1:
            continue
        assert len(deps_two) == 1, f"last node has multiple?\n{node_one}\n{node_two}\n{deps_two}" 
        take_idx = list(deps_two)[0] 

        # in nodes_two, get what nodes to keep and take if it depends on main node 
        take_node = node_two.nodes[take_idx] 
        take_res = get_res(take_node)
        nodes_to_take = []
        nodes_to_keep = []
        for idx, node in enumerate(node_two.nodes):
            if len(get_deps(node).intersection(take_res)) > 0 or idx == take_idx:
                nodes_to_take.append(node)
            else:
                nodes_to_keep.append(node)
        node_two.nodes = nodes_to_keep
        id_to_node[dep_two] = node_two
        res = get_res(node_two) 

        # remove dependency
        g_dep[dep_two].remove(dep_one)

        # insert all nodes in g_dep 
        for n in nodes_to_take:
           
            # insert node
            l = g_dep.setdefault(n.id, set())
            n_res = get_deps(n)
            for path_id in path:
                path_res = get_res(id_to_node[path_id])
                if len(n_res.intersection(path_res)) > 0:
                    l.add(path_id)

            #l.add(dep_one)
            #l.add(dep_two)
            
            # assert that this node is in id_to_node
            id_to_node[n.id] = n

            # if there's a dependency on the original node, fix
            for node_dep in g_dep:
                if dep_two in deepcopy(g_dep[node_dep]):
                    if n.id == node_dep: continue
                    
                    # 4 possiblities
                    # 1. need both (just add n.id)
                    # 2. need original fused node (remove n.id), 
                    # 3. need new node (remove search, add n.id)
                    # 4. None. In this case, it will be handled by othern nodes_to_take!

                    deps = get_deps(id_to_node[node_dep])
                    need_orig = len(deps.intersection(res)) > 0
                    need_new = len(deps.intersection({n.id})) > 0 

                    if need_new and need_orig:
                        g_dep[node_dep].add(n.id)
                    elif need_orig:
                        if n.id in g_dep[node_dep]:
                            g_dep[node_dep].remove(n.id)
                    elif need_new:
                        g_dep[node_dep].remove(dep_two)
                        g_dep[node_dep].add(n.id)


