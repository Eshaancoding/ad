from autodiff.graph.data.feeder import Feeder
from . import *
from math import prod
from typing import Dict, Set, Tuple
from ..fusion import FuseBase
from ..fusion.helper import get_deps, get_res
from pprint import pprint 

# returning fusebase as it's helpful for the future alloc opt: fusion temp
def insert_alloc (main_proc: Proc) -> Dict[int, FuseBase]:
    # key: [idx from proc, fuse_idx for fusebase (if applicable. Else none)]
    # value: idx to insert
    insert_locs: Dict[int, Tuple[AllocEntry, Location]] = {}
    dealloc_locs: Dict[int, Tuple[Location, DeallocEntry]] = {}
    tracker: Dict[int, bool] = {}
    
    fused_ids_to_f: Dict[int, FuseBase] = {}
    proc_ids_to_p: Dict[int, Proc] = {}
    id_to_size: Dict[int, int] = {}
    
    def step_proc (proc: Proc, proc_id=None):
        for idx, p in enumerate(proc.procedure):
            def step_node (loc: Location, n: Node):
                # returning fusebase as it's helpful for the future alloc opt: fusion temp
                assert isinstance(n, Node)

                # record res allocations 
                r = list(get_res(n))[0]

                if r in insert_locs:
                    insert_locs[r][0].size = max(insert_locs[r][1].size, prod(n.shape)) # if reused, ensure max shape 
                elif r in id_to_size:
                    # declared in a tensor if not declared in insert_locs but declared in id_to_size
                    assert id_to_size[r] == prod(n.shape), "In place operation dimension are not equal when inserting alloc"
                else:
                    insert_locs[r] = (loc, AllocEntry(r, prod(n.shape)))

                tracker[n.id] = False
                
                id_to_size[n.id] = prod(n.shape)
                
                # record dealloc
                deps = get_deps(n)
                for dep in list(deps):
                    dealloc_locs[dep] = (
                        loc+1,
                        DeallocEntry(dep, id_to_size[dep])
                    )
                    tracker[dep] = True
        
            if isinstance(p, FuseBase):
                for fuse_idx, n in enumerate(p.nodes):
                    fused_ids_to_f[p.fuse_id] = p
                    if isinstance(n, Tensor):
                        p.nodes[fuse_idx] = AllocEntry(n.id, prod(n.shape), n.arr)
                        id_to_size[n.id] = prod(n.shape)
                        continue

                    step_node(
                        Location(fuse_idx, fused_id=p.fuse_id, proc_id=None),
                        n
                    )
            elif isinstance(p, Node):
                if isinstance(p, Tensor):
                    proc.procedure[idx] = AllocEntry(p.id, prod(p.shape), p.arr)
                    id_to_size[p.id] = prod(p.shape)
                    continue
                
                if p.get_proc() is not None:
                    proc_ids_to_p[proc.id] = p.proc
                    step_proc(p.proc, proc.id) 
                    continue

                step_node(
                    Location(idx, fused_id=None, proc_id=proc_id),
                    p
                )
            else:
                raise TypeError(f"Invalid type: {type(p)}")
                
    step_proc(main_proc)
    
    tracker = dict(filter(lambda kv: not kv[1], tracker.items()))
    tracker = tracker.keys()

    # insert leftover variables at the end
    # TODO: Include dep opt to make this happen less
    # + repeat opt (and general node optimization)
    for var in tracker:
        dealloc_locs[var] = (
            Location(len(main_proc.procedure), None, None),
            DeallocEntry(var, id_to_size[var])
        )

    # now, actually insert alloc
    insert_counter: Dict[FuseBase, int] = {}
    total = list(dealloc_locs.values())
    total.extend(insert_locs.values())
    total = sorted(total, key=lambda x: (
        s if (s := x[0].proc_id) is not None else -1,
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
        elif loc.proc_id is not None:
            proc_ids_to_p[loc.proc_id].insert(loc.loc + offset, entry)
        else:
            main_proc.insert(loc.loc + offset, entry)
             
        # incr 
        insert_counter[base] += 1
        
    return fused_ids_to_f
