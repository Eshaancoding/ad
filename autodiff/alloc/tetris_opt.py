from . import *
from ..alloc import AllocEntry, DeallocEntry
from typing import Dict
from pprint import pprint
from dataclasses import dataclass
from ..expr import *

@dataclass
class VarEntry:
    var_id: str
    start_idx: int
    end_idx: Optional[int]
    size: int
    offset: int=0

idx = 0

def plot (var_entries: List[VarEntry]):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot each rectangle
    for entry in var_entries:
        width = entry.size
        height = entry.end_idx - entry.start_idx
        rect = patches.Rectangle(
            (entry.offset, entry.start_idx),
            width,
            height,
            linewidth=1,
            edgecolor='blue',
            facecolor='cyan',
            alpha=0.5
        )
        ax.add_patch(rect)
        # Label with var_id
        ax.text(entry.offset + width / 2, (entry.start_idx + entry.end_idx) / 2, str(entry.var_id),
                ha='center', va='center', fontsize=8)

    # Set axis labels and limits
    ax.set_xlabel("Offset (x)")
    ax.set_ylabel("Index Range (y)")
    ax.set_title("Variable Layout Visualization")
    ax.autoscale_view()

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def overlap_locs (a: VarEntry, b: VarEntry):
    start = max(a.start_idx, b.start_idx)
    end = min(a.end_idx, b.end_idx)
    return start < end

def ranges_intersect(r1, r2):
    """Return True if two ranges (start1, end1) and (start2, end2) intersect."""
    return r1[0] < r2[1] and r2[0] < r1[1]

def merge_ranges(r1, r2):
    """Merge two overlapping ranges into one. Assumes they intersect."""
    return (min(r1[0], r2[0]), max(r1[1], r2[1]))

def tetris_opt (proc: Proc):
    global idx

    var_tracker: Dict[int, Dict[int, VarEntry]] = {}

    def walk_proc (node: Node|AllocCmds, proc_id: str):
        global idx
        if proc_id not in var_tracker:
            var_tracker[proc_id] = {}

        if isinstance(node, AllocEntry):
            if not node.is_temp:
                var_tracker[proc_id][node.id] = VarEntry(node.id, idx, None, node.size)
        elif isinstance(node, DeallocEntry):
            if node.id in var_tracker[proc_id] and not node.is_temp:
                var_tracker[proc_id][node.id].end_idx = idx
        idx += 1 

        return node

    idx = 0
    proc.walk(walk_proc)

    # remove dict values where there's None values
    for proc_id in var_tracker:
        var_tracker[proc_id] = {k:v for k, v in var_tracker[proc_id].items() if v.end_idx is not None}
    
    # x.end_idx - x.start_idx = 153792
    # -x.size = 153702
    # 41504

    var_tracker: Dict[int, List[VarEntry]] = {
        proc_id : list(sorted(v.values(), key=lambda x: -x.size))
        for proc_id, v in var_tracker.items() if len(v) > 0
    } # filter out empty dicts

    # figure out offsets  
    max_temp_size = 0
    total_size = 0
    offsetted_entry: Dict[int, List[VarEntry]] = {}
    for proc_id in var_tracker:
        offsetted_entry[proc_id] = []
    
        for entry in var_tracker[proc_id]:
            total_size += entry.size

            taken = []
            for o_entry in offsetted_entry[proc_id]:
                if overlap_locs(entry, o_entry):
                    taken.append((o_entry.offset, o_entry.offset + o_entry.size))

            taken = sorted(taken, key=lambda x: x[0])

            # then, make sure that in taken there's no overlap 
            idx_r = 0
            while idx_r < len(taken)-1:
                range_one = taken[idx_r]
                range_two = taken[idx_r+1]
                if ranges_intersect(range_one, range_two):
                    new_range = merge_ranges(range_one, range_two)
                    del taken[idx_r]
                    del taken[idx_r]
                    taken.insert(idx_r, new_range) 
                    #taken = sorted(taken, key=lambda x: x[0])
                else:
                    idx_r += 1

            # see if we can fill in between gaps
            offset = -1
            for idx_t in range(len(taken)-1): 
                gap_size = taken[idx_t+1][0] - taken[idx_t][1]
                if gap_size >= entry.size:
                    offset = taken[idx_t][1]

            if offset == -1:
                offset = taken[-1][-1] if len(taken) > 0 else 0

            entry.offset = offset
            offsetted_entry[proc_id].append(entry)
            max_temp_size = max(max_temp_size, offset + entry.size)

    temp_id = context.get_id()

    # debug
    if False:
        print("================== Alloc Debug ==================")
        print(f"Using: {max_temp_size} out of {total_size}")
        print(f"Saved {int(total_size - max_temp_size)}. Using {(max_temp_size/total_size)*100:.3f}%")
        plot(offsetted_entry[0])

    # then, replace the values needed
    def replace_to_temp (node, proc_id:int):
        if proc_id not in offsetted_entry:
            return node 

        entries = offsetted_entry[proc_id]
        if isinstance(node, AllocEntry) or isinstance(node, DeallocEntry):
            for entry in entries:
                if entry.var_id == node.id:
                    return None

        if not isinstance(node, Node):
            return node

        for idx in range(len(node.kargs)):
            for entry in entries:
                def add_expr (expr: Expression):
                    if entry.offset == 0:
                        return expr
                    else:
                        return Add(expr, Val(Constant(entry.offset)))
                node.kargs[idx].rename(entry.var_id, temp_id, add_expr) 
                node.kres.rename(entry.var_id, temp_id, add_expr)

        return node
    
    proc.walk(replace_to_temp)

    proc.insert(0, AllocEntry(temp_id, max_temp_size, content=None))
    proc.append(DeallocEntry(temp_id, max_temp_size))
