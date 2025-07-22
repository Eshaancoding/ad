from . import *
from math import prod
from typing import Dict, Tuple
from pprint import pprint

# TODO: Make sure this supports blocks within blocks (if, for, etc.)
# returning fusebase as it's helpful for the future alloc opt: fusion temp
def insert_alloc (proc: List[Node | FuseBase]) -> Dict[int, FuseBase]:
    # find the allocations and deallocations locations at List[Node] and insert them
    # NOTE: Doesn't support fusion within a fusion

    # key: [idx from proc, fuse_idx for fusebase (if applicable. Else none)]
    # value: idx to insert
    insert_locs: List[Tuple[Location, AllocEntry]] = []
    dealloc_locs: Dict[str, Tuple[Location, DeallocEntry]] = {}
    
    fused_ids_to_f: Dict[int, FuseBase] = {}
    
    id_to_size: Dict[int, int] = {}
    
    for idx, p in enumerate(proc):
        def step_node (loc: Location, n: Node):
            # returning fusebase as it's helpful for the future alloc opt: fusion temp
            assert isinstance(n, Node)

            # record res allocations 
            r = list(get_res(n))[0]
            
            insert_locs.append((
                loc,
                AllocEntry(n.id, prod(n.shape))
            ))
            
            id_to_size[n.id] = prod(n.shape)
            
            # record dealloc
            deps = get_deps(n)
            for dep in list(deps):
                dealloc_locs[dep] = (
                    loc+1,
                    DeallocEntry(dep, id_to_size[dep])
                )
       
        if isinstance(p, FuseBase):
            for fuse_idx, n in enumerate(p.nodes):
                fused_ids_to_f[p.fuse_id] = p
                if isinstance(n, Tensor):
                    p.nodes[fuse_idx] = AllocEntry(n.id, prod(n.shape), n.data)
                    id_to_size[n.id] = prod(n.shape)
                    continue

                step_node(
                    Location(fuse_idx, fused_id=p.fuse_id, control_node=None),
                    n
                )
        else:
            if isinstance(p, Tensor):
                proc[idx] = AllocEntry(p.id, prod(p.shape), p.data)
                id_to_size[p.id] = prod(p.shape)
                continue

            step_node(
                Location(idx, fused_id=None, control_node=None),
                p
            )

    # now, actually insert alloc
    insert_counter: Dict[FuseBase, int] = {}
    total = list(dealloc_locs.values())
    total.extend(insert_locs)
    total = sorted(total, key=lambda x: (
        s if (s := x[0].control_node) is not None else -1,
        s if (s := x[0].fused_id) is not None else -1,
        x[0].loc
    ))    

    for (loc, entry) in total:
        # get insert counter
        base = loc.base()
        offset = 0
        if base in insert_counter:
            offset = insert_counter[base]
        else:
            insert_counter[base] = 0
           
        # insert
        if loc.fused_id is not None:
            fused_ids_to_f[loc.fused_id].nodes.insert(loc.loc + offset, entry)
        else:
            proc.insert(loc.loc + offset, entry)
             
        # incr 
        insert_counter[base] += 1
        
    return fused_ids_to_f