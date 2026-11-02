"""cpsa.helpers — composable physics and object-abstraction primitives.

Supports both integer color IDs (0..15) and ARC color chars ('W'..'p'),
auto-detects background color instead of hardcoding 0, and provides
`objects_of(frame)` for writing compact boundary projections `project(frame)`.
"""

import numpy as np
from .wmlib import color_char, resolve_color
from .world_model import _extract_grid

DIRS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


def bg_color(frame):
    """Return the dominant (background) integer color ID of the board."""
    grid = _extract_grid(frame)
    u, counts = np.unique(grid, return_counts=True)
    return int(u[np.argmax(counts)]) if len(u) else 0


def find_color(frame, color):
    """[(r, c)] of all cells with color (int 0..15 or ARC char 'W'..'p')."""
    grid = _extract_grid(frame)
    c = resolve_color(color)
    ys, xs = np.nonzero(grid == c)
    return [(int(r), int(col)) for r, col in zip(ys, xs)]


def first_of(frame, color):
    """First (r, c) of color in scan order, or None."""
    pts = find_color(frame, color)
    return pts[0] if pts else None


def objects_of(frame, include_bg=False):
    """Extract 4-connected same-color components as a list of dicts:
    [{'color': int, 'char': str, 'pixels': int, 'bbox': (r0,c0,r1,c1),
      'topleft': (r0,c0), 'center': (cr,cc), 'cells': [(r,c), ...]}].
    Ideal for building `project(frame)` state abstractions."""
    grid = _extract_grid(frame)
    H, W = grid.shape
    bg = None if include_bg else bg_color(grid)
    seen = np.zeros((H, W), dtype=bool)
    objs = []
    for r0 in range(H):
        for c0 in range(W):
            if seen[r0, c0]:
                continue
            val = int(grid[r0, c0])
            if bg is not None and val == bg:
                seen[r0, c0] = True
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == val:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comp.sort()
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            objs.append({
                "color": val,
                "char": color_char(val),
                "pixels": len(comp),
                "bbox": (min(rs), min(cs), max(rs), max(cs)),
                "topleft": (min(rs), min(cs)),
                "center": (sum(rs) // len(comp), sum(cs) // len(comp)),
                "cells": comp,
            })
    return objs


def in_bounds(frame, r, c):
    h, w = _extract_grid(frame).shape
    return 0 <= r < h and 0 <= c < w


def move_object(frame, color, dr, dc, walls=(), push=None, wrap=False, bg=None):
    """Move all cells of `color` by (dr, dc), supporting multi-cell avatars and
    multi-cell pushed boxes (Sokoban-style connected components).

    walls: iterable of colors (int or ARC char) that block movement.
    push:  color (int or ARC char) pushed one step further (Sokoban-style).
    wrap:  wrap around board edges instead of blocking.
    bg:    background fill color for vacated cells (auto-detected if None).
    """
    c_val = resolve_color(color)
    wall_set = tuple(resolve_color(w) for w in (walls or ()))
    push_val = resolve_color(push) if push is not None else None
    pts = find_color(frame, c_val)
    if not pts:
        return False
    pts_set = set(pts)
    h, w = frame.grid.shape
    if bg is None:
        fill_bg = 0 if int(np.count_nonzero(frame.grid == 0)) >= (h * w) // 4 else bg_color(frame)
    else:
        fill_bg = resolve_color(bg)

    dest = []
    pushed_pts = set()
    for (r, c) in pts:
        nr, nc = r + dr, c + dc
        if wrap:
            nr %= h
            nc %= w
        if not in_bounds(frame, nr, nc):
            return False
        occupant = int(frame.grid[nr, nc])
        if occupant in wall_set:
            return False
        if push_val is not None and occupant == push_val:
            # Collect the full 4-connected component of push_val touched at (nr, nc)
            stack = [(nr, nc)]
            while stack:
                br, bc = stack.pop()
                if (br, bc) in pushed_pts:
                    continue
                pushed_pts.add((br, bc))
                for pdr, pdc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nbr, nbc = br + pdr, bc + pdc
                    if in_bounds(frame, nbr, nbc) and int(frame.grid[nbr, nbc]) == push_val and (nbr, nbc) not in pushed_pts:
                        stack.append((nbr, nbc))
        elif occupant != fill_bg and (nr, nc) not in pts_set and occupant != c_val:
            # Non-background, non-pushable obstacle if not explicitly in walls
            pass
        dest.append((nr, nc))

    # Validate destination of every pushed pixel
    pushed_dest = []
    for (br, bc) in pushed_pts:
        pr, pc = br + dr, bc + dc
        if wrap:
            pr %= h
            pc %= w
        if not in_bounds(frame, pr, pc):
            return False
        occ = int(frame.grid[pr, pc])
        if occ in wall_set:
            return False
        if (pr, pc) not in pushed_pts and (pr, pc) not in pts_set and occ in (c_val, push_val):
            return False
        pushed_dest.append((pr, pc))

    for (br, bc) in pushed_pts:
        frame.grid[br, bc] = fill_bg
    for (r, c) in pts:
        frame.grid[r, c] = fill_bg
    for (pr, pc) in pushed_dest:
        frame.grid[pr, pc] = push_val
    for (nr, nc) in dest:
        frame.grid[nr, nc] = c_val
    return True


def move_by_action(frame, color, walls=(), push=None, wrap=False, bg=None):
    """Parse DIRS from frame.state['_action'] and move `color`."""
    act = str(frame.state.get("_action", "")).upper()
    if act not in DIRS:
        return False
    dr, dc = DIRS[act]
    return move_object(frame, color, dr, dc, walls=walls, push=push, wrap=wrap, bg=bg)


def tick_budget(frame, key="budget", step=-1, floor=None):
    val = int(frame.state.get(key, 0)) + int(step)
    if floor is not None:
        val = max(int(floor), val)
    frame.state[key] = val
    return val


def phase_step(frame, key, period):
    val = (int(frame.state.get(key, 0)) + 1) % max(1, int(period))
    frame.state[key] = val
    return val


def paint(frame, cells_rc, color):
    c_val = resolve_color(color)
    for (r, c) in cells_rc:
        if in_bounds(frame, r, c):
            frame.grid[r, c] = c_val


def copy_region(frame, src_bbox, dst_rc):
    r0, c0, r1, c1 = src_bbox
    dr, dc = dst_rc
    block = frame.grid[r0:r1 + 1, c0:c1 + 1].copy()
    frame.grid[dr:dr + block.shape[0], dc:dc + block.shape[1]] = block


def manhattan(frame, color_a, color_b):
    """Manhattan distance between first cells of two colors (A* heuristic)."""
    a = first_of(frame, color_a)
    b = first_of(frame, color_b)
    if a is None or b is None:
        return 0.0
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


SIM_SKELETON = '''
# Option A: Full-grid simulator (SimFrame 64x64)
def simulate(frame, action):
    f = frame.copy()
    f.state["_action"] = action
    # Compose primitives (colors accept int 0..15 or ARC char 'W'..'p'):
    # moved = move_by_action(f, AVATAR_COLOR, walls=(WALL_COLOR,), push=BOX_COLOR)
    return f

def is_goal(frame):
    # Derive from wmlib.win_transition() / pre_level_swap_grid.
    return False

# Option B (faster & simpler on multi-tile boards): Boundary Projection pi_K
# def project(frame):
#     return {"avatar": first_of(frame, AVATAR_COLOR), "boxes": tuple(find_color(frame, BOX_COLOR))}
# def simulate(state, action):
#     st = copy_state(state)
#     ...
#     return st
# def is_goal(state):
#     return st["avatar"] == TARGET_RC
'''


def make_sim_skeleton():
    """Print and return a starting template for simulate()/is_goal()."""
    print(SIM_SKELETON)
    return SIM_SKELETON


HELPER_EXPORTS = {
    "bg_color": bg_color,
    "find_color": find_color,
    "first_of": first_of,
    "objects_of": objects_of,
    "in_bounds": in_bounds,
    "move_object": move_object,
    "move_by_action": move_by_action,
    "tick_budget": tick_budget,
    "phase_step": phase_step,
    "paint": paint,
    "copy_region": copy_region,
    "manhattan": manhattan,
    "make_sim_skeleton": make_sim_skeleton,
    "DIRS": DIRS,
}
