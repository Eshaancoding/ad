# inside fusion operations, if there's an alloc and dealloc inside a fused operation, then only used temp variables
# we keep the allocs and deallocs there! We will remove the dealloc at a future function
# Will be part of clean
# 1. take dealloc without temp out of fusion operations of fusion operations of fusion operations
# 2. remove temp deallocs as already implicit via C++
# 3. etc.

from . import *
from ..fusion import FuseBase
from typing import Dict

def temp_alloc(proc: Proc, fused_ids_to_of: Dict[int, FuseBase]):
    for bases in fused_ids_to_of.values():
        # for each base, track the nodes that have both dealloc and allloc
        track = {}
        for n in bases.nodes:
            if isinstance(n, AllocEntry):
                track[n.id] = False
            elif isinstance(n, DeallocEntry):
                if n.id in track:
                    track[n.id] = True

        track = set([val for val, key in track.items() if key == True])

        for n in bases.nodes:
            if isinstance(n, AllocEntry):
                if n.id in track:
                    n.is_temp = True
            if isinstance(n, DeallocEntry):
                if n.id in track:
                    n.is_temp = True
            if isinstance(n, Node):
                for t in track:
                    n.change_to_temp(t)
