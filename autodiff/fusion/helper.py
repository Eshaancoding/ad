from .base import Node
from typing import Set

def get_deps (node) -> Set[str]:
    from .base import FuseBase
    if isinstance(node, Node):
        return set(node.kargs_child_ids())
    elif isinstance(node, FuseBase):
        r = []
        for n in node.nodes:
            r.extend(get_deps(n))
        return set(r)
    else:
        raise TypeError(f"Invalid type in calc_deps: {type(node)}")
    
def get_res (node) -> Set[str]:
    from .base import FuseBase
    if isinstance(node, Node):
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
        remove = []
        
        for id_one in dupl[id_two]:
            r = get_res(id_to_node[id_one])
            
            for res_id in r: 
                if res_id not in deps:
                    remove.append(id_one)
        
        remove = list(set(remove))
        
        # if remove list is equal to the total list, then remove one
        if len(remove) == len(dupl[id_two]):
            remove.pop()
            
            # debug
            if debug:
                raise Exception(f"Multiple many to one matches")
        
        for r in remove:
            del matches[r]
    
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