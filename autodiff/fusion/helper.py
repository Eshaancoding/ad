from .base import FuseBase, Node, AllocEntry
from typing import Set, List, Dict
from toposort import toposort

def get_deps (node: Node|FuseBase|AllocEntry, step_proc:bool=False) -> Set[int]:
    from .base import FuseBase
    if isinstance(node, Node):
        if step_proc and (block := node.get_proc()) is not None:
            ret = []
            for n in block.procedure:
                ret.extend(get_deps(n, True))
            return set(ret) 
        else:
            return set(node.kargs_child_ids())
    elif isinstance(node, FuseBase):
        r = []
        for n in node.nodes:
            r.extend(get_deps(n))
        return set(r)
    else:
        raise TypeError(f"Invalid type in calc_deps: {type(node)}")

def print_toposort (toposort_res, id_to_node):
    for idx, layer in enumerate(toposort_res):
        print(f"\n================ Layer #{idx+1} ================")
        for node in layer:
            print(id_to_node[node]) 
    
def get_res (node: Node|FuseBase|AllocEntry, step_proc:bool=False) -> Set[int]:
    from .base import FuseBase
    if isinstance(node, Node):
        if step_proc and (block := node.get_proc()) is not None:
            ret = []
            for n in block.procedure:
                ret.extend(get_deps(n, True))
            return set(ret) 
        else:
            return set([node.id])
    elif isinstance(node, FuseBase):
        r = []
        for n in node.nodes:
            r.extend(get_res(n))
        return set(r)
    else:
        raise TypeError(f"Invalid type in calc_res: {type(node)}")

def resolve_one_to_many (matches, id_to_node, debug=False):
    for id_one in matches:
        if len(matches[id_one]) == 1:
            matches[id_one] = matches[id_one][0] # flatten list
        else:
            # get res id 
            node_one = id_to_node[id_one]
            res = get_res(node_one)

            # find deps with that res id
            id_select = None
            max_intersect = None
            for id_two in matches[id_one]:
                node_two = id_to_node[id_two]
                deps = get_deps(node_two)
                if max_intersect is None or len(deps.intersection(res)) > max_intersect:
                    max_intersect = len(deps.intersection(res))
                    id_select = id_two
                    
            if id_select is None:
                st = ""
                for id_two in matches[id_one]:
                    node_two = id_to_node[id_two]
                    st += f"{node_two} with des: {get_deps(node_two)}\n"
                raise Exception(f"Multiple fuse matches for node:\n{node_one}\n\nOptions: \n{st}\n\nRes: {res}")
            
            # set matches (chose the first id if invalid)
            matches[id_one] = id_select if id_select is not None else matches[id_one][0]
            
            # debug
            if id_select is None and debug:
                print(f"Multiple fuse matches for node: {node_one}")

    return matches

def resolve_many_to_one (matches, id_to_node, debug=False):
    dupl = {} 
    for id_one in matches:
        id_two = matches[id_one] 
        if id_two not in dupl:
            dupl[id_two] = [id_one]
        else:
            dupl[id_two].append(id_one)
            
    dupl = {key:value for key,value in dupl.items() if len(value) > 1}

    for id_two in dupl:
        deps = get_deps(id_to_node[id_two]) 
        keep = set()
        
        for id_one in dupl[id_two]:
            r = get_res(id_to_node[id_one])
            if len(r.intersection(deps)) > 0:
                keep.add(id_one)
        
        # if we are keeping NOTHING, then debug 
        if len(keep) == 0:
            if debug:
                print(f"******* Many to one match = none matches res ******")

        # if we are keeping more than one value, then keep nothing (invalid fuse 
        if len(keep) > 1:
            keep = []
            if debug:
                print(f"******* INVALID FUSE! MANY TO ONE ******")
                
        for id_one in dupl[id_two]:
            if id_one not in keep:
                del matches[id_one]

    return matches

def resolve_circular_dep (matches):
    # Handles cases like
    # {1: 2, 2: 1}
    seen = set()
    new_matches = {}    
    for key, value in matches.items():
        if (value, key) in seen:
            continue
        new_matches[key] = value
        seen.add((key, value)) 
   
    # Handles cases like
    # {1: 2, 2: 3}
    keys = set(new_matches.keys())
    values = set(new_matches.values())
    
    l = keys.intersection(values)
    
    for k in l:
        del new_matches[k]
   
    return new_matches 

def clean_toposort_res (matches, id_to_node):
    # iterate over toposort, and filter sets that are 0 
    matches = list(filter(lambda x: len(x) > 0, matches)) 

    # for each layer, get all the definitions of the layer
    definitions = []
    for layer in matches:
        res = []
        for node in layer:
            res.extend(get_res(id_to_node[node], step_proc=True))
        res = set(res) # unique res
        definitions.append(res)
    for idx in reversed(range(len(matches))):
        if idx == 0: 
            continue

        # track what should be moved
        move = []
        for node in matches[idx]:
            n_deps = get_deps(id_to_node[node], step_proc=True)
            if definitions[idx-1].intersection(n_deps):
                continue
            move.append(node)

        # move to previous layer
        for node in move:
            matches[idx].remove(node)
            matches[idx-1].add(node)

    # iterate over toposort, and filter sets that are 0 
    matches = list(filter(lambda x: len(x) > 0, matches)) 
    
    return matches
            
def flatten_toposort (toposort_res, id_to_node, already_decl):
    # Recalculate toposort
    g_dep: Dict[int, List[int]] = {}
    declared_in_fused = {}
    for layer in toposort_res:
        for id in layer:
            node = id_to_node[id]
            if isinstance(node, FuseBase):
                g_dep[node.fuse_id] = list(get_deps(node, True))
                for n in node.nodes:
                    r = get_res(n)
                    assert len(r) == 1, "Len res 1 invalid"
                    r = list(r)[0]
                    # if already declared, then we don't need to store to res_to_fused
                    if r not in already_decl:
                        declared_in_fused[r] = node.fuse_id
            elif isinstance(node, Node):
                g_dep[node.id] = list(get_deps(node, True))

    for n_id in g_dep:
        # if ids in fuse_id, replace
        new_l = set()
        for n in g_dep[n_id]:
            to_insert = declared_in_fused[n] if n in declared_in_fused else n
            if to_insert != n_id:
                new_l.add(to_insert)
        
        # remove ids in already decl
        g_dep[n_id] = list(filter(lambda item: item not in already_decl, new_l))
    g_dep = {key:val for key, val in g_dep.items() if len(val) > 0}

    # recalculate toposort_res
    toposort_res = list(toposort(g_dep))
    
    # then, we can flatten
    nodes = []
    for layer in toposort_res:
        for id in layer:
            if id in id_to_node:
                # some ids might not be included (as it's already declared inside a fused node
                # note that this is not the best setup... try to simplify this 
                nodes.append(id_to_node[id])
    return nodes
