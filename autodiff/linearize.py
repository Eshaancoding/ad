from copy import deepcopy
from typing import Dict, List
from autodiff.context import Block, Proc
from autodiff.fusion.base import FuseBase, FuseResult
from autodiff.fusion.helper import get_deps, get_res, get_walking_path, print_toposort
from autodiff.fusion.ops import ElwFuse, DPElwFuse, ReduceElwFuse
from autodiff.helper import walk_graph
from autodiff.node import Node
from autodiff.print_graph import pg
from toposort import toposort
from pprint import pprint

global_deps = {}
id_to_node = {}
block_to_node = {}
alr_defined_global = {}

def insert_to_global_dep (block_id:int, key:int, value:int):
    global global_deps
    if block_id not in global_deps:
        global_deps[block_id] = dict() # initialize global deps with empty dictionary 

    if key not in global_deps[block_id]:
        if value is not None:
            global_deps[block_id][key] = set({value}) 
        else:
            global_deps[block_id][key] = set({}) 
    elif value is not None:
        global_deps[block_id][key].add(value)

def get_deps_global (block: Block, alr_defined: Dict[int, int] = dict()):
    global block_to_node, alr_defined_global
   
    # track alr defined 
    if block.id not in alr_defined_global:
        alr_defined_global[block.id] = list(alr_defined.keys())

    defined = dict()
    vars_need_dep = dict()
    
    for n in block.nodes:
        # step through the block
        if (inner_bl := n.get_block()):
            def_block, need_dep_block = get_deps_global(inner_bl, alr_defined | defined)
        
            for (dep, block_id) in deepcopy(list(need_dep_block.items())):
                if block_id == block.id:
                    insert_to_global_dep(block.id, n.id, dep)
                    del need_dep_block[dep]

            # update variables
            defined.update(def_block)
            vars_need_dep.update(need_dep_block)
            block_to_node[inner_bl.id] = n.id
        else:
            # if regular node, go through the ids
            visited = set()
            def update_node (n:Node, visited):
                # skip if node if already defined
                if n.id in visited:
                    return n
                if n.id in alr_defined:
                    return n

                if isinstance(n, Node):
                    # insert to global deps
                    for dep in n.kargs_child_ids():
                        should_visit = True 
                        key = n.id
                        value = dep

                        # referencing node already defined by lower-level blocks
                        # ex: for loop referencing tensor defined in global block
                        if dep in alr_defined:
                            value = None
                            should_visit = False
                            vars_need_dep[dep] = alr_defined[dep]
                        
                        # referencing node already defined by upper-level blocks
                        # ex: Receiver of the latest weight changed by upper for loop 
                        if dep in defined:
                            if (block_id_defined := defined[dep]) != block.id:
                                should_visit = False
                                value = block_to_node[block_id_defined]
                                
                        # insert to global dep
                        insert_to_global_dep(block.id, key, value)

                        # visit
                        if should_visit:
                            visited.add(n.id)
                            update_node(id_to_node[dep], visited)

                    # add to defined if not inserted already 
                    if n.id not in defined:
                        defined[n.id] = block.id
                return n
            
            update_node(n, visited)

    return defined, vars_need_dep

def fuse (g_dep_id) -> Proc:
    ########### Run through toposort ########### 
    global global_deps, alr_defined_global
    deps = global_deps[g_dep_id]

    # run through toposort
    toposort_res = list(toposort(deps))
    #print_toposort(toposort_res, id_to_node)
    #if g_dep_id == 1:
    #    exit(0)

    ########### Fusing Operation ###########
    ops = [
        DPElwFuse,
        ReduceElwFuse,
        ElwFuse  
    ]

    entries: List[FuseBase] = []
    init_taken = set()

    for op in ops:
        while True:
            op_entry: FuseBase = op() 
            white_list_nodes = set()
            
            # Fusion process
            did_start = False
            for layer_idx, layer in enumerate(toposort_res):
                did_sum = False

                for node_idx in layer:
                    deps_id = get_deps(id_to_node[node_idx])

                    if node_idx in init_taken:
                        # add to white listing if nodes intersect 
                        if len(deps_id.intersection(op_entry.result_tracker)) > 0:
                            white_list_nodes.update(get_walking_path(
                                node=id_to_node[node_idx],
                                toposort_res=toposort_res[layer_idx+1:],
                                id_to_node=id_to_node
                            ))
                        continue # then skip 
                    if node_idx in white_list_nodes:
                        continue

                    # attempt fuseMulti
                    # alr defined doesn't matter
                    result = op_entry.attempt_fuse(id_to_node[node_idx])

                    if result in (FuseResult.ADD, FuseResult.INIT):
                        init_taken.add(node_idx)
                        did_sum = True
                        did_start = True
                    elif len(deps_id.intersection(op_entry.result_tracker)) > 0:
                        # add to whitelisting
                        white_list_nodes.update(get_walking_path(
                            node=id_to_node[node_idx],
                            toposort_res=toposort_res[layer_idx+1:],
                            id_to_node=id_to_node
                        ))

                if did_start and not did_sum:
                    break

            # Fuse more of the node or carry on to the next
            if len(op_entry.nodes) == 0: 
                break # can't fuse no more
            if len(op_entry.nodes) == 1:
                init_taken.add(op_entry.nodes[0].id)
            elif len(op_entry.nodes) > 1:
                entries.append(op_entry)

    # insert the fuse entries + get node to remove
    nodes_to_remove = set()
    for entry in entries:
        did_found = False
        for idx, layer in enumerate(toposort_res):
            for id_node in layer:
                if entry.nodes[-1].id == id_node:
                    id_to_node[entry.fuse_id] = entry
                    toposort_res[idx].add(entry.fuse_id)
                    did_found = True
                    break
            if did_found:
                break
        if not did_found:
            raise Exception("Couldn't find node in toposort_res to insert fused entry")

        for node in entry.nodes:
            nodes_to_remove.add(node.id)

    # then, remove all in toposort res
    for idx, layer in enumerate(toposort_res):
        toposort_res[idx] = set(filter(lambda x: x not in nodes_to_remove, layer))

    ####### Fill dep_to_fuse_id + run fuse on inner blocks ####### 

    # replace dependency with their fuse id
    replace = dict() 
    for layer in toposort_res:
        for node_id in layer:
            node = id_to_node[node_id]

            # if we have a node with block (ex: for), then fuse that as well
            # should already exist in g_deps
            if isinstance(node, Node) and (block := node.get_block()) is not None:
                node.proc = fuse(block.id)
                replace.update({r: node.id for r in get_res(node, True)})

            # dep to fuse id
            if isinstance(node, FuseBase):
                replace.update({r: node.fuse_id for r in get_res(node)})

    ########### Rerun toposort ###########
    g_dep = {} 
    for layer in toposort_res:
        for idx in layer: 
            node = id_to_node[idx]
            deps = get_deps(node, step_proc=True)

            # replace 
            deps = set({
                (replace[d] if d in replace else d)
                for d in deps
            })

            # then get g_deps
            id = node.id if isinstance(node, Node) else node.fuse_id
            g_dep[id] = deps

    # filter deps to only include the keys defined (everything else has been defined later)
    for id in g_dep:
        g_dep[id] = set(filter(lambda x: x in g_dep.keys(), g_dep[id]))

    # finally, toposort it
    toposort_res = list(toposort(g_dep))

    ########### Linearize --> Return to proc ###########
    proc = []
    for layer in toposort_res:
        for node_id in layer:
            node = id_to_node[node_id]

            # else, return to proc
            proc.append(node)
        
    return Proc(proc)

def linearize (main_block: Block):
    global global_deps, id_to_node, block_to_node, alr_defined_global

    ######## Insert all id to node ######## 
    id_to_node = {}
    def insert_id_to_node (node: Node, _):
        if node.id not in id_to_node:
            id_to_node[node.id] = node
        return node
    walk_graph(main_block, insert_id_to_node, walk_block=True, walk_child=True) 

    ######## Get deps ########
    global_deps = {}
    block_to_node = {}
    alr_defined_global = {}
    _, d = get_deps_global(main_block)
    assert len(d) == 0, "vars need dep not None!"

    ######## Given the deps: fuse ########
    return fuse(0) # fuse the global node + return proc 
