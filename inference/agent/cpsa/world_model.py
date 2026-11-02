"""cpsa.world_model — SimFrame, Boundary Projection (pi_K), V4 Judge, search,
and guarded execution.

Self-contained: stdlib + numpy + cpsa.wmlib only. No __future__ imports.
"""

import os
import heapq
import inspect
import numpy as np
from collections import deque

from .wmlib import (
    ARC_COLOR_CHARS, TransitionRecord, color_char, diff_text, resolve_color,
)

UNKNOWN_CELL = -1

_COLOR_CHARS = ".123456789ABCDEF"

ADVISORY_ACC = float(os.environ.get("ARC3_CPSA_ADVISORY_ACC", "0.85"))
ADVISORY_COV = float(os.environ.get("ARC3_CPSA_ADVISORY_COV", "0.75"))
MAX_UNKNOWN_RATIO = 0.15


def _canonicalize_state(state):
    if state is None:
        return ()
    if isinstance(state, np.ndarray):
        return ("ndarray", state.shape, state.tobytes())
    if isinstance(state, (int, float, str, bool, bytes)):
        return state
    if isinstance(state, (list, tuple)):
        return tuple(_canonicalize_state(x) for x in state)
    if isinstance(state, (set, frozenset)):
        return tuple(sorted((_canonicalize_state(x) for x in state), key=str))
    if isinstance(state, dict):
        items = []
        for k in sorted(state.keys(), key=str):
            items.append((k, _canonicalize_state(state[k])))
        return tuple(items)
    return str(state)


def copy_state(state):
    if state is None:
        return {}
    if isinstance(state, np.ndarray):
        return state.copy()
    if isinstance(state, (int, float, str, bool, bytes, tuple, frozenset)):
        return state
    if isinstance(state, list):
        return [copy_state(x) for x in state]
    if isinstance(state, set):
        return {copy_state(x) for x in state}
    if isinstance(state, dict):
        return {k: copy_state(v) for k, v in state.items()}
    return state


def _extract_grid(grid_like):
    """Extract a 2D numpy int16 array from SimFrame, FrameView (.grid or ._grid),
    ndarray, or nested sequence."""
    if isinstance(grid_like, SimFrame):
        return grid_like.grid
    if hasattr(grid_like, "grid"):
        g = getattr(grid_like, "grid")
        if g is not None and not callable(g):
            return np.asarray(g, dtype=np.int16)
    if hasattr(grid_like, "_grid"):
        g = getattr(grid_like, "_grid")
        if g is not None and not callable(g):
            return np.asarray(g, dtype=np.int16)
    return np.asarray(grid_like, dtype=np.int16)


class SimFrame:
    """Moore-machine state: 64x64 grid (frame.grid) + hidden vars (frame.state).

    Assigning None to a cell marks it UNKNOWN_CELL (-1) = local abstention.
    Accepts FrameView (via .grid or ._grid), SimFrame, ndarray, or nested lists.
    """

    def __init__(self, grid, state=None, dead=False):
        if isinstance(grid, SimFrame):
            self.grid = grid.grid.copy()
            self.state = copy_state(state if state is not None else grid.state)
            self.dead = bool(dead or getattr(grid, "dead", False))
            return
        if hasattr(grid, "grid") or hasattr(grid, "_grid"):
            arr_in = _extract_grid(grid)
            self.grid = arr_in.astype(np.int16, copy=True)
            self.state = copy_state(state) if state is not None else {}
            self.dead = bool(dead)
            return
        if isinstance(grid, np.ndarray):
            self.grid = grid.astype(np.int16, copy=True)
            self.state = copy_state(state) if state is not None else {}
            self.dead = bool(dead)
            return
        h = len(grid)
        w = len(grid[0]) if h else 0
        arr = np.empty((h, w), dtype=np.int16)
        for r in range(h):
            for c in range(w):
                val = grid[r][c]
                arr[r, c] = UNKNOWN_CELL if val is None else resolve_color(val)
        self.grid = arr
        self.state = copy_state(state) if state is not None else {}
        self.dead = bool(dead)

    def copy(self):
        return SimFrame(self.grid.copy(), copy_state(self.state), dead=self.dead)

    @property
    def shape(self):
        return self.grid.shape

    def __getitem__(self, idx):
        return self.grid[idx]

    def __setitem__(self, idx, val):
        if val is None:
            self.grid[idx] = UNKNOWN_CELL
        elif isinstance(val, str):
            self.grid[idx] = resolve_color(val)
        else:
            self.grid[idx] = val

    def find(self, color):
        c = resolve_color(color)
        ys, xs = np.nonzero(self.grid == c)
        return [(int(r), int( col)) for r, col in zip(ys, xs)]

    def bbox(self, color):
        pts = self.find(color)
        if not pts:
            return None
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        return (min(rs), min(cs), max(rs), max(cs))


class FalsifiedGoalLedger:
    """Immutable blacklist of states where is_goal() returned True but the
    environment did NOT complete the level. search() refuses blacklisted goals
    at zero environment cost."""

    def __init__(self):
        self._falsified_frames = {}

    def record_false_positive_goal(self, level, grid, state, seq_id, is_goal_fn):
        try:
            src = inspect.getsource(is_goal_fn).strip()
        except Exception:
            src = getattr(is_goal_fn, "__name__", "<lambda>")
        bucket = self._falsified_frames.setdefault(int(level), [])
        g_copy = np.array(_extract_grid(grid), dtype=np.int8, copy=True)
        canon = _canonicalize_state(state)
        for existing_g, existing_st, _, _ in bucket:
            if np.array_equal(existing_g, g_copy) and _canonicalize_state(existing_st) == canon:
                return
        bucket.append((g_copy, copy_state(state), int(seq_id), src))

    def validate_goal_candidate(self, level, is_goal_fn, project=None):
        bucket = self._falsified_frames.get(int(level), [])
        for idx, (g_false, st_false, seq_id, _) in enumerate(bucket):
            probe = SimFrame(g_false, st_false)
            try:
                arg = project(probe) if project is not None else probe
                res = bool(is_goal_fn(arg))
            except Exception as e:
                return False, f"is_goal raised {type(e).__name__}: {e} on falsified frame #{idx + 1}"
            if res:
                return (
                    False,
                    f"REJECTED BY JUDGE: Your `is_goal` still returns True on a state already "
                    f"reached at transition #{seq_id} where the level DID NOT COMPLETE! "
                    f"You MUST modify `is_goal` so it returns False on that state before calling search().",
                )
        return True, None

    def to_dict(self):
        out = {}
        for lvl, entries in self._falsified_frames.items():
            out[str(lvl)] = [
                {"grid": g.tolist(), "state": _jsonable(st), "seq_id": seq, "src": src}
                for g, st, seq, src in entries
            ]
        return out

    def load_dict(self, data):
        self._falsified_frames.clear()
        if not data:
            return
        for lvl_str, entries in data.items():
            bucket = []
            for item in entries:
                bucket.append((
                    np.array(item["grid"], dtype=np.int8),
                    _state_from_json(item.get("state")),
                    int(item["seq_id"]),
                    str(item.get("src", "")),
                ))
            self._falsified_frames[int(lvl_str)] = bucket


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return {"__ndarray__": v.tolist()}
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, tuple):
        return {"__tuple__": [_jsonable(x) for x in v]}
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (set, frozenset)):
        return {"__set__": [_jsonable(x) for x in sorted(v, key=str)]}
    return v


def _state_from_json(v):
    if isinstance(v, dict):
        if "__ndarray__" in v:
            return np.array(v["__ndarray__"])
        if "__tuple__" in v:
            return tuple(_state_from_json(x) for x in v["__tuple__"])
        if "__set__" in v:
            return set(_state_from_json(x) for x in v["__set__"])
        return {k: _state_from_json(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_state_from_json(x) for x in v]
    return v


def _observable_projection(st):
    """Extract the observable quotient part of a projected state `st`.
    When `st` is a dict, keys starting with '_' (e.g. '_phase', '_latent', '_dead')
    carry hidden Moore-machine memory across steps without failing comparison
    against memoryless `project(SimFrame(rec.after))`.
    """
    if isinstance(st, dict):
        return {k: v for k, v in st.items() if not str(k).startswith("_")}
    return st


def _is_dead_state(f_or_st):
    """Return True if a SimFrame or projected state represents a fatal/hazard state."""
    if f_or_st is None:
        return False
    if isinstance(f_or_st, SimFrame):
        if getattr(f_or_st, "dead", False):
            return True
        if isinstance(f_or_st.state, dict):
            return bool(
                f_or_st.state.get("dead")
                or f_or_st.state.get("game_over")
                or f_or_st.state.get("_dead")
            )
        return False
    if isinstance(f_or_st, dict):
        return bool(
            f_or_st.get("dead")
            or f_or_st.get("game_over")
            or f_or_st.get("_dead")
        )
    return False


# ---------------------------------------------------------------- expectation
# predicates for Mode 1 (act_expect)

def expect_moved(color, dr=None, dc=None):
    c = resolve_color(color)

    def _check(prev_f, curr_f, act, diff_coords):
        b0 = prev_f.bbox(c)
        b1 = curr_f.bbox(c)
        if b0 is None or b1 is None:
            return False, f"Color {color} disappeared or was not found (before={b0}, after={b1})."
        actual_dr = b1[0] - b0[0]
        actual_dc = b1[1] - b0[1]
        if dr is not None and actual_dr != dr:
            return False, f"Color {color} row delta was {actual_dr}, expected {dr} on action {act}."
        if dc is not None and actual_dc != dc:
            return False, f"Color {color} col delta was {actual_dc}, expected {dc} on action {act}."
        if dr is None and dc is None and (actual_dr == 0 and actual_dc == 0):
            return False, f"Color {color} did not move at all on action {act} (bbox stayed {b0})."
        return True, f"Color {color} moved by ({actual_dr}, {actual_dc})."
    return _check


def expect_region_changed(bbox_rect):
    r0, c0, r1, c1 = bbox_rect

    def _check(prev_f, curr_f, act, diff_coords):
        inside = [(r, c, o, n) for (r, c, o, n) in diff_coords if r0 <= r <= r1 and c0 <= c <= c1]
        if not inside:
            return False, f"No cells changed inside region {bbox_rect} on action {act}."
        return True, f"{len(inside)} cells changed inside region {bbox_rect}."
    return _check


def expect_diff_count(min_changes=1, max_changes=64):
    def _check(prev_f, curr_f, act, diff_coords):
        cnt = len(diff_coords)
        if cnt < min_changes or cnt > max_changes:
            return False, (
                f"Changed cell count {cnt} outside expected range "
                f"[{min_changes}, {max_changes}] on action {act}."
            )
        return True, f"Changed cell count {cnt} within [{min_changes}, {max_changes}]."
    return _check


def expect_cells(changes, allow_extra=False):
    """Exact per-cell expectation: changes is iterable of (r, c, new_color)."""
    want = {(int(r), int(c)): resolve_color(v) for (r, c, v) in changes}

    def _check(prev_f, curr_f, act, diff_coords):
        for (r, c), v in want.items():
            got = int(curr_f.grid[r, c])
            if got != v:
                return False, f"Cell ({r},{c}) expected color {v} after {act}, got {got}."
        if not allow_extra:
            actual = {(r, c): n for (r, c, o, n) in diff_coords}
            extra = {k: v for k, v in actual.items() if k not in want or v != want[k]}
            if extra:
                sample = sorted(extra.items())[:6]
                return False, (
                    f"Unexpected extra changes on {act}: "
                    f"{[(k[0], k[1], v) for k, v in sample]} "
                    f"({len(extra)} cells; pass allow_extra=True to permit)."
                )
        return True, f"All {len(want)} expected cells match."
    return _check


# ------------------------------------------------------------------ inspection
# utilities (compatible with FrameView, SimFrame, ndarray, and ARC chars)

def cells(grid, color):
    """[(r, c)] of every cell equal to color (int 0..15 or ARC char)."""
    arr = _extract_grid(grid)
    c = resolve_color(color)
    return [(int(r), int(col)) for r, col in np.argwhere(arr == c)]


def bbox(pts_or_color, grid=None):
    """(min_r, min_c, max_r, max_c) of a point list, or of a color within grid."""
    if grid is not None:
        pts = cells(grid, pts_or_color)
    else:
        pts = pts_or_color
    pts = list(pts or [])
    if not pts:
        return None
    rs = [int(p[0]) for p in pts]
    cs = [int(p[1]) for p in pts]
    return (min(rs), min(cs), max(rs), max(cs))


def colors(grid):
    """{color_id: count} histogram of the whole board."""
    arr = _extract_grid(grid)
    u, counts = np.unique(arr, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, counts)}


SHOW_WINDOW_MAX_SIDE = 32


def show_window(grid, r0=0, c0=0, r1=None, c1=None, color_chars=_COLOR_CHARS):
    """Compact symbolic crop with coordinate rulers. Windows larger than
    SHOW_WINDOW_MAX_SIDE per side are refused: never dump the full 64x64 board."""
    arr = _extract_grid(grid)
    H, W = arr.shape
    r1v = H - 1 if r1 is None else min(H - 1, int(r1))
    c1v = W - 1 if c1 is None else min(W - 1, int(c1))
    r0v, c0v = max(0, int(r0)), max(0, int(c0))
    if r1v - r0v + 1 > SHOW_WINDOW_MAX_SIDE or c1v - c0v + 1 > SHOW_WINDOW_MAX_SIDE:
        raise RuntimeError(
            f"show_window: region {r1v - r0v + 1}x{c1v - c0v + 1} exceeds "
            f"{SHOW_WINDOW_MAX_SIDE}x{SHOW_WINDOW_MAX_SIDE}. Inspect a smaller crop, "
            f"or use colors()/cells()/bbox()/current_frame.segmentation for the whole board."
        )
    lines = ["     " + "".join(str(c % 10) for c in range(c0v, c1v + 1))]
    for r in range(r0v, r1v + 1):
        row = []
        for c in range(c0v, c1v + 1):
            val = int(arr[r, c])
            if val == UNKNOWN_CELL:
                row.append("?")
            elif color_chars == _COLOR_CHARS and val == 0:
                row.append(".")
            elif 0 <= val < len(color_chars):
                row.append(color_chars[val])
            else:
                row.append(f"{val % 16:X}")
        lines.append(f"{r:>4} {''.join(row)}")
    res = "\n".join(lines)
    print(res)
    return res


def candidate_clicks(frame, max_clicks=64, bg=None):
    """Focused MOUSE-click candidates for search() in clicking games.

    Returns action strings 'MOUSE(row=R, col=C)' for connected-component
    centroids of every non-background color. Auto-detects the background color
    by majority area when bg is None (instead of hardcoding 0).
    """
    grid = _extract_grid(frame)
    H, W = grid.shape
    if bg is None:
        u, counts = np.unique(grid, return_counts=True)
        bg_val = int(u[np.argmax(counts)]) if len(u) else 0
    else:
        bg_val = resolve_color(bg)
    seen = np.zeros((H, W), dtype=bool)
    pts = []
    for r0 in range(H):
        for c0 in range(W):
            if seen[r0, c0] or int(grid[r0, c0]) == bg_val:
                continue
            color = int(grid[r0, c0])
            stack = [(r0, c0)]
            comp = []
            seen[r0, c0] = True
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            sr = sum(p[0] for p in comp) // len(comp)
            sc = sum(p[1] for p in comp) // len(comp)
            if (sr, sc) not in comp:
                sr, sc = min(comp, key=lambda p: abs(p[0] - sr) + abs(p[1] - sc))
            sr = max(0, min(H - 1, int(sr)))
            sc = max(0, min(W - 1, int(sc)))
            is_interactive_size = 1 if (1 <= len(comp) <= 128) else 0
            pts.append((is_interactive_size, len(comp), sr, sc))
    # Prioritize interactive-sized objects (1..128 px) before large borders/walls (>128 px)
    pts.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    out = []
    for _, _, sr, sc in pts[:max_clicks]:
        s = f"MOUSE(row={sr}, col={sc})"
        if s not in out:
            out.append(s)
    return out


# ----------------------------------------------------------------------- Judge
# (Supports both Grid/SimFrame mode and Boundary Projection pi_K Quotient mode,
# with V4 verdict accounting: FOLLOW / OMIT / UNDECIDED / NULL)

def _is_trivial_false_goal(is_goal_fn):
    """Return True if `is_goal_fn` is a constant-False/None stub (Axiom 4.1 MBEL
    & NonTriv(y) guard: prevents SIM_SKELETON's `return False` from getting PASS
    on levels without a recorded win_transition)."""
    if not callable(is_goal_fn):
        return True
    try:
        import ast
        import textwrap
        src = textwrap.dedent(inspect.getsource(is_goal_fn)).strip()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda):
                return isinstance(node.body, ast.Constant) and not bool(node.body.value)
            if isinstance(node, ast.FunctionDef):
                returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
                if not returns:
                    return True
                return all(
                    (r.value is None)
                    or (isinstance(r.value, ast.Constant) and not bool(r.value.value))
                    for r in returns
                )
    except Exception:
        pass
    try:
        import dis
        code = getattr(is_goal_fn, "__code__", None)
        if code is not None:
            ins = list(dis.get_instructions(code))
            active_ops = {
                i.opname for i in ins
                if i.opname not in {
                    "RESUME", "NOP", "LOAD_CONST", "RETURN_VALUE",
                    "RETURN_CONST", "POP_TOP", "COPY_FREE_VARS",
                }
            }
            if not active_ops:
                ret_vals = [
                    i.argval for i in ins
                    if i.opname in {"LOAD_CONST", "RETURN_CONST"}
                ]
                if not ret_vals or not bool(ret_vals[-1]):
                    return True
    except Exception:
        pass
    return False


def _finalize_operator_stats(op_stats, win_rec=None, goal_ok=False):
    """Compute MBEL §3.4 per-operator confirmation status chi_conf(a).
    An operator `a` has chi_conf(a) = 1 iff it has >= 1 FOLLOW transition and
    0 NULL/UNDECIDED transitions on the evaluated ensemble.
    Operators with OMIT > 0 and FOLLOW == 0 are flagged as unconfirmed_noop_operators.
    """
    stats = {k: dict(v) for k, v in op_stats.items()}
    if win_rec is not None and goal_ok:
        slot = stats.setdefault(win_rec.action, {"FOLLOW": 0, "OMIT": 0, "UNDECIDED": 0, "NULL": 0})
        slot["FOLLOW"] += 1
    confirmed = sorted(
        a for a, s in stats.items()
        if s["FOLLOW"] > 0 and s["NULL"] == 0 and s["UNDECIDED"] == 0
    )
    unconfirmed_noop = sorted(
        a for a, s in stats.items()
        if s["FOLLOW"] == 0 and s["OMIT"] > 0 and s["NULL"] == 0 and s["UNDECIDED"] == 0
    )
    return stats, confirmed, unconfirmed_noop


def _evaluate_projected_model(wmlib, falsified_ledger, simulate, is_goal, project, lvl, hud_mask):
    recs = wmlib.transitions(level=lvl, include_game_over=False, include_level_completed=False)
    win_rec = wmlib.win_transition(lvl)
    result = {
        "ok": False, "advisory_ok": False, "simulation_accuracy": None,
        "active_accuracy": None,
        "prediction_coverage": None, "known_cell_accuracy": None,
        "passed": 0, "total": 0, "failures": [], "goal_violations": [],
        "first_divergence": None, "per_level": {}, "ledger_rejection": None,
        "v4_counts": {"FOLLOW": 0, "OMIT": 0, "UNDECIDED": 0, "NULL": 0},
        "operator_stats": {},
        "confirmed_operators": [],
        "unconfirmed_noop_operators": [],
        "goal_status": "UNDECIDED",
        "mode": "PROJECTED",
        "report": "",
    }
    if is_goal is not None:
        if _is_trivial_false_goal(is_goal):
            result["goal_violations"].append(
                "VACUOUS GOAL VIOLATION (Axiom 4.1 MBEL / NonTriv(y)): "
                "is_goal() is a constant-False stub. Define a non-trivial goal "
                "condition before certifying the model or calling search()."
            )
            result["goal_status"] = "CONTRADICTED"
        valid_goal, goal_err = falsified_ledger.validate_goal_candidate(lvl, is_goal, project=project)
        if not valid_goal and goal_err:
            result["ledger_rejection"] = goal_err
            result["goal_violations"].append(goal_err)
            result["goal_status"] = "CONTRADICTED"

    attempts = {}
    for r in recs:
        attempts.setdefault(r.attempt, []).append(r)

    passed = total = 0
    active_total = active_captured = 0
    failures = []
    goal_violations = list(result["goal_violations"])
    first_div = None
    non_hud = ~hud_mask
    last_state_by_attempt = {}

    for att_id, att_recs in sorted(attempts.items()):
        s_threaded = None
        for rec in att_recs:
            total += 1
            actual_changed = bool(np.any((rec.before != rec.after) & non_hud))
            try:
                s_before_obs = project(SimFrame(rec.before))
                s_after_obs = project(SimFrame(rec.after))
                if (
                    s_threaded is not None
                    and _canonicalize_state(_observable_projection(s_threaded))
                    == _canonicalize_state(_observable_projection(s_before_obs))
                ):
                    s_in = s_threaded
                else:
                    s_in = s_before_obs
                s_pred = simulate(copy_state(s_in), rec.action)
            except Exception as e:
                result["v4_counts"]["NULL"] += 1
                s_threaded = None
                failures.append(
                    f"Attempt {att_id}, Step {rec.step_in_attempt} ({rec.action}): "
                    f"projected model raised {type(e).__name__}: {e}"
                )
                if first_div is None:
                    first_div = {
                        "level": rec.level, "attempt": att_id,
                        "step": rec.step_in_attempt, "action": rec.action,
                        "observed": None, "predicted": None,
                        "error": f"{type(e).__name__}: {e}",
                    }
                continue

            if s_pred is None:
                result["v4_counts"]["UNDECIDED"] += 1
                s_threaded = None
                failures.append(
                    f"Attempt {att_id}, Step {rec.step_in_attempt} ({rec.action}): "
                    f"projected simulate() returned None (UNDECIDED)."
                )
                continue

            s_threaded = copy_state(s_pred)
            last_state_by_attempt[att_id] = copy_state(s_pred)

            c_before = _canonicalize_state(_observable_projection(s_before_obs))
            c_after = _canonicalize_state(_observable_projection(s_after_obs))
            c_pred = _canonicalize_state(_observable_projection(s_pred))

            if actual_changed:
                active_total += 1
                if c_before != c_after:
                    active_captured += 1

            op_slot = result["operator_stats"].setdefault(
                rec.action, {"FOLLOW": 0, "OMIT": 0, "UNDECIDED": 0, "NULL": 0}
            )
            if c_pred != c_after:
                result["v4_counts"]["NULL"] += 1
                op_slot["NULL"] += 1
                step_fail = (
                    f"Attempt {att_id}, Step {rec.step_in_attempt} (seq #{rec.seq_id}, "
                    f"action={rec.action}): projected state mismatch:\n"
                    f"  expected project(after) = {_observable_projection(s_after_obs)!r}\n"
                    f"  simulate(state, {rec.action!r}) = {_observable_projection(s_pred)!r}"
                )
                failures.append(step_fail)
                if first_div is None:
                    first_div = {
                        "level": rec.level, "attempt": att_id,
                        "step": rec.step_in_attempt, "action": rec.action,
                        "observed": repr(_observable_projection(s_after_obs)),
                        "predicted": repr(_observable_projection(s_pred)),
                    }
            else:
                passed += 1
                if actual_changed and c_before != c_after:
                    result["v4_counts"]["FOLLOW"] += 1
                    op_slot["FOLLOW"] += 1
                elif not actual_changed:
                    result["v4_counts"]["OMIT"] += 1
                    op_slot["OMIT"] += 1
                else:
                    result["v4_counts"]["UNDECIDED"] += 1
                    op_slot["UNDECIDED"] += 1

            if is_goal is not None:
                try:
                    goal_arg = s_threaded if c_pred == c_after and s_threaded is not None else s_after_obs
                    if bool(is_goal(copy_state(goal_arg))):
                        goal_violations.append(
                            f"GOAL VIOLATION at Attempt {att_id}, Step {rec.step_in_attempt} "
                            f"(seq #{rec.seq_id}): is_goal(project(frame)) returned True on a "
                            f"non-completing historical frame!"
                        )
                except Exception as e:
                    goal_violations.append(f"is_goal raised {type(e).__name__}: {e}")

    # Kernel inclusion check: ker(project) <= ker(env)
    coverage = (active_captured / active_total) if active_total > 0 else 1.0
    if active_total > 0 and coverage < ADVISORY_COV:
        failures.append(
            f"KERNEL INCLUSION VIOLATION (ker(project) not in ker(env)): "
            f"project(frame) only distinguished {active_captured}/{active_total} "
            f"({coverage:.0%} < {ADVISORY_COV:.0%}) transitions where non-HUD cells "
            f"changed! Include all moving gameplay objects in project(frame)."
        )

    if win_rec is not None and is_goal is not None:
        try:
            s_win_before_obs = project(SimFrame(win_rec.before))
            att_st = last_state_by_attempt.get(win_rec.attempt)
            if (
                att_st is not None
                and _canonicalize_state(_observable_projection(att_st))
                == _canonicalize_state(_observable_projection(s_win_before_obs))
            ):
                s_win_in = att_st
            else:
                s_win_in = s_win_before_obs
            s_win_pred = simulate(copy_state(s_win_in), win_rec.action)
            goal_on_pred = bool(is_goal(copy_state(s_win_pred))) if s_win_pred is not None else False
            if not goal_on_pred:
                goal_violations.append(
                    f"WINNING TRANSITION VIOLATION (seq #{win_rec.seq_id}, action={win_rec.action}): "
                    f"is_goal(simulate(project(win.before), win.action)) returned False on the winning step!"
                )
            elif not goal_violations and result["ledger_rejection"] is None:
                result["goal_status"] = "CONFIRMED"
        except Exception as e:
            goal_violations.append(f"Winning step check raised {type(e).__name__}: {e}")

    if goal_violations or result["ledger_rejection"] is not None:
        result["goal_status"] = "CONTRADICTED"

    has_follow = (result["v4_counts"]["FOLLOW"] > 0) or (win_rec is not None)
    # Cross-level transfer fallback (Bug #15): if lvl > 1 has 0 FOLLOW transitions and 0 failures,
    # check if the model is certified on the most recent completed prior level.
    if not has_follow and lvl > 1 and not failures and not goal_violations and result["ledger_rejection"] is None:
        for prev_lvl in sorted({r.level for r in wmlib._records if r.level < lvl}, reverse=True):
            if wmlib.transitions(level=prev_lvl) or wmlib.win_transition(prev_lvl) is not None:
                prev_res = _evaluate_projected_model(
                    wmlib, falsified_ledger, simulate, None, project,
                    prev_lvl, wmlib.compute_hud_mask(prev_lvl),
                )
                if prev_res["ok"] or prev_res["advisory_ok"]:
                    sf = wmlib.level_start_grid(lvl)
                    if is_goal is not None and sf is not None:
                        try:
                            if bool(is_goal(copy_state(project(SimFrame(sf))))):
                                break
                        except Exception:
                            break
                    prev_res["report"] = (
                        f"[check_sim/PROJECTED] Level {lvl} has 0 active steps; "
                        f"physics certified on Level {prev_lvl}, is_goal validated on Level {lvl}:\n"
                        + prev_res["report"]
                    )
                    return prev_res
                break

    if not has_follow:
        failures.append(
            "VACUOUS MODEL VIOLATION (Axiom 4.1 MBEL): FOLLOW=0 — projected "
            "simulator has not positively explained any active transition on this level."
        )

    accuracy = (passed / total) if total else None
    v4 = result["v4_counts"]
    active_denom = v4["FOLLOW"] + v4["UNDECIDED"] + v4["NULL"]
    active_acc = (v4["FOLLOW"] / active_denom) if active_denom > 0 else accuracy
    kernel_ok = (active_total == 0) or (coverage >= ADVISORY_COV)
    ok = (
        (passed == total)
        and has_follow
        and kernel_ok
        and (not goal_violations)
        and (result["ledger_rejection"] is None)
    )
    advisory_ok = (
        not ok
        and has_follow
        and kernel_ok
        and result["ledger_rejection"] is None
        and not goal_violations
        and accuracy is not None
        and accuracy >= ADVISORY_ACC
        and active_acc is not None
        and active_acc >= ADVISORY_ACC
    )
    op_stats, conf_ops, unconf_ops = _finalize_operator_stats(
        result["operator_stats"], win_rec=win_rec, goal_ok=(result["goal_status"] == "CONFIRMED" or is_goal is None)
    )
    result.update({
        "ok": ok, "advisory_ok": advisory_ok,
        "simulation_accuracy": accuracy, "active_accuracy": active_acc,
        "prediction_coverage": coverage,
        "known_cell_accuracy": accuracy,
        "operator_stats": op_stats,
        "confirmed_operators": conf_ops,
        "unconfirmed_noop_operators": unconf_ops,
        "passed": passed, "total": total,
        "failures": failures[:8], "goal_violations": goal_violations[:5],
        "first_divergence": first_div,
        "per_level": {lvl: {"accuracy": accuracy, "passed": passed, "total": total,
                            "first_divergence": first_div}},
    })
    lines = [
        f"[check_sim/PROJECTED] Level {lvl}: {passed}/{total} transitions matched"
        + (f" (accuracy={accuracy:.2f}, active_acc={active_acc:.2f}, kernel_cov={coverage:.2f})" if accuracy is not None else "")
        + f" | V4: FOLLOW={v4['FOLLOW']} OMIT={v4['OMIT']} U={v4['UNDECIDED']} NULL={v4['NULL']}"
        + f" | goal={result['goal_status']}."
    ]
    if failures:
        lines.append("PHYSICS / KERNEL MISMATCHES:")
        lines.extend("  - " + f for f in failures[:8])
    if goal_violations:
        lines.append("GOAL INVARIANT VIOLATIONS:")
        lines.extend("  - " + g for g in goal_violations[:5])
    if ok:
        lines.append("VERDICT: PASS — projected simulator and goal are 100% consistent with observed history.")
    elif advisory_ok:
        lines.append(
            f"VERDICT: ADVISORY-OK (accuracy={accuracy:.2f}, active_acc={active_acc:.2f} >= {ADVISORY_ACC:.2f}) — "
            f"search() is allowed; act_checked will guard every step."
        )
    else:
        lines.append("VERDICT: FAIL — fix the reported mismatches before search() is allowed.")
    result["report"] = "\n".join(lines)
    return result


def evaluate_model(wmlib, falsified_ledger, simulate, is_goal=None, level=None,
                   max_unknown_ratio=MAX_UNKNOWN_RATIO, project=None):
    """Pure verification core (the Deterministic V4 Judge). No printing, no IPC.

    When `project` is None, runs in Grid/SimFrame mode (64x64).
    When `project(frame) -> state` is supplied, runs in Boundary Projection
    (pi_K quotient) mode, checking commutativity and kernel inclusion
    ker(project) <= ker(env).
    """
    lvl = wmlib.current_level if level is None else int(level)
    recs = wmlib.transitions(level=lvl, include_game_over=False, include_level_completed=False)
    win_rec = wmlib.win_transition(lvl)
    result = {
        "ok": False, "advisory_ok": False, "simulation_accuracy": None,
        "active_accuracy": None,
        "prediction_coverage": None, "known_cell_accuracy": None,
        "passed": 0, "total": 0, "failures": [], "goal_violations": [],
        "first_divergence": None, "per_level": {}, "ledger_rejection": None,
        "v4_counts": {"FOLLOW": 0, "OMIT": 0, "UNDECIDED": 0, "NULL": 0},
        "operator_stats": {},
        "confirmed_operators": [],
        "unconfirmed_noop_operators": [],
        "goal_status": "UNDECIDED",
        "mode": "GRID",
        "report": "",
    }
    if not recs and win_rec is None:
        # Cross-level transfer on Turn 0 of Level 2+ (Bug #15)
        if lvl > 1:
            if is_goal is not None:
                if _is_trivial_false_goal(is_goal):
                    result["goal_violations"].append(
                        "VACUOUS GOAL VIOLATION (Axiom 4.1 MBEL / NonTriv(y)): "
                        "is_goal() is a constant-False stub."
                    )
                    result["goal_status"] = "CONTRADICTED"
                    result["report"] = "\n".join(result["goal_violations"])
                    return result
                valid_goal, goal_err = falsified_ledger.validate_goal_candidate(lvl, is_goal, project=project)
                if not valid_goal and goal_err:
                    result["ledger_rejection"] = goal_err
                    result["goal_violations"].append(goal_err)
                    result["goal_status"] = "CONTRADICTED"
                    result["report"] = goal_err
                    return result
            for prev_lvl in sorted({r.level for r in wmlib._records if r.level < lvl}, reverse=True):
                if wmlib.transitions(level=prev_lvl) or wmlib.win_transition(prev_lvl) is not None:
                    prev_res = evaluate_model(
                        wmlib, falsified_ledger, simulate, is_goal=None,
                        level=prev_lvl, max_unknown_ratio=max_unknown_ratio, project=project,
                    )
                    if prev_res["ok"] or prev_res["advisory_ok"]:
                        sf = wmlib.level_start_grid(lvl)
                        if is_goal is not None and sf is not None:
                            try:
                                arg0 = copy_state(project(SimFrame(sf))) if project is not None else SimFrame(sf)
                                if bool(is_goal(arg0)):
                                    result["goal_violations"].append(
                                        f"GOAL VIOLATION on Level {lvl}: is_goal() returned True on initial start grid!"
                                    )
                                    result["goal_status"] = "CONTRADICTED"
                                    result["report"] = "\n".join(result["goal_violations"])
                                    return result
                            except Exception as e:
                                result["goal_violations"].append(f"is_goal raised {type(e).__name__}: {e}")
                                result["goal_status"] = "CONTRADICTED"
                                result["report"] = "\n".join(result["goal_violations"])
                                return result
                        prev_res["report"] = (
                            f"[check_sim] Level {lvl} Turn 0 (0 transitions on level {lvl}): "
                            f"physics certified on Level {prev_lvl}, is_goal validated on Level {lvl}.\n"
                            + prev_res["report"]
                        )
                        return prev_res
                    break
        result["report"] = (
            f"[check_sim] WARNING: 0 transitions recorded on level {lvl}. "
            f"Take at least 1 action first!"
        )
        return result

    hud_mask = wmlib.compute_hud_mask(lvl)

    if project is not None:
        return _evaluate_projected_model(
            wmlib, falsified_ledger, simulate, is_goal, project, lvl, hud_mask
        )

    if is_goal is not None:
        if _is_trivial_false_goal(is_goal):
            result["goal_violations"].append(
                "VACUOUS GOAL VIOLATION (Axiom 4.1 MBEL / NonTriv(y)): "
                "is_goal() is a constant-False stub. Define a non-trivial goal "
                "condition before certifying the model or calling search()."
            )
            result["goal_status"] = "CONTRADICTED"
        valid_goal, goal_err = falsified_ledger.validate_goal_candidate(lvl, is_goal)
        if not valid_goal and goal_err:
            result["ledger_rejection"] = goal_err
            result["goal_violations"].append(goal_err)
            result["goal_status"] = "CONTRADICTED"

    attempts = {}
    for r in recs:
        attempts.setdefault(r.attempt, []).append(r)

    passed = total = 0
    failures = []
    goal_violations = list(result["goal_violations"])
    first_div = None
    cov_sum = cov_n = 0
    known_ok = known_n = 0
    last_state_by_attempt = {}

    for att_id, att_recs in sorted(attempts.items()):
        sim_state = {}
        for rec in att_recs:
            total += 1
            sim_f = SimFrame(rec.before, state=sim_state)
            try:
                pred_out = simulate(sim_f.copy(), rec.action)
            except Exception as e:
                result["v4_counts"]["NULL"] += 1
                failures.append(
                    f"Attempt {att_id}, Step {rec.step_in_attempt} ({rec.action}): "
                    f"simulate() raised {type(e).__name__}: {e}"
                )
                if first_div is None:
                    first_div = {
                        "level": rec.level, "attempt": att_id,
                        "step": rec.step_in_attempt, "action": rec.action,
                        "observed": None, "predicted": None,
                        "error": f"{type(e).__name__}: {e}",
                    }
                continue
            if pred_out is None:
                result["v4_counts"]["UNDECIDED"] += 1
                failures.append(
                    f"Attempt {att_id}, Step {rec.step_in_attempt} ({rec.action}): "
                    f"simulate() returned None (UNDECIDED step-level abstention)."
                )
                if first_div is None:
                    first_div = {
                        "level": rec.level, "attempt": att_id,
                        "step": rec.step_in_attempt, "action": rec.action,
                        "observed": diff_text(rec.before, rec.after),
                        "predicted": "None (UNDECIDED)",
                    }
                continue
            pred_f = pred_out if isinstance(pred_out, SimFrame) else SimFrame(pred_out, sim_f.state)
            sim_state = copy_state(pred_f.state)
            last_state_by_attempt[att_id] = copy_state(sim_state)

            pred = pred_f.grid
            non_hud = ~hud_mask
            unknown_mask = (pred == UNKNOWN_CELL) & non_hud
            unknown_ratio = float(np.count_nonzero(unknown_mask)) / float(64 * 64)
            claimed = non_hud & (pred != UNKNOWN_CELL)
            cov_sum += float(np.count_nonzero(claimed)) / float(np.count_nonzero(non_hud))
            cov_n += 1
            kmask = claimed & (pred == rec.after.astype(np.int16))
            known_ok += int(np.count_nonzero(kmask))
            known_n += int(np.count_nonzero(claimed))

            actual_changed = (rec.before != rec.after) & non_hud
            has_real_change = bool(np.any(actual_changed))

            step_fail = None
            step_undecided = False
            if unknown_ratio > max_unknown_ratio:
                step_fail = (
                    f"Attempt {att_id}, Step {rec.step_in_attempt} ({rec.action}): "
                    f"too many None/UNKNOWN cells ({unknown_ratio:.1%} > "
                    f"{max_unknown_ratio:.0%})."
                )
            else:
                unexplained = actual_changed & unknown_mask
                if np.any(unexplained):
                    ur, uc = np.nonzero(unexplained)
                    r0, c0 = int(ur[0]), int(uc[0])
                    step_undecided = True
                    step_fail = (
                        f"Attempt {att_id}, Step {rec.step_in_attempt} ({rec.action}): "
                        f"UNEXPLAINED CHANGE — cell ({r0},{c0}) really changed "
                        f"{int(rec.before[r0, c0])}->{int(rec.after[r0, c0])} but simulate() "
                        f"returned None there. Every real change must be predicted with a color."
                    )
                else:
                    mismatches = claimed & (pred != rec.after.astype(np.int16))
                    if np.any(mismatches):
                        actual_masked = rec.after.astype(np.int16).copy()
                        pred_masked = pred.copy()
                        ignore = ~claimed
                        actual_masked[ignore] = 0
                        pred_masked[ignore] = 0
                        compact = diff_text(actual_masked, pred_masked)
                        n_mis = int(np.count_nonzero(mismatches))
                        step_fail = (
                            f"Attempt {att_id}, Step {rec.step_in_attempt} (seq #{rec.seq_id}, "
                            f"action={rec.action}): {n_mis} cell mismatch(es); "
                            f"actual-vs-predicted diff:\n{compact}"
                        )
                        if first_div is None:
                            first_div = {
                                "level": rec.level, "attempt": att_id,
                                "step": rec.step_in_attempt, "action": rec.action,
                                "observed": diff_text(rec.before, rec.after),
                                "predicted": diff_text(rec.before, pred),
                            }
            op_slot = result["operator_stats"].setdefault(
                rec.action, {"FOLLOW": 0, "OMIT": 0, "UNDECIDED": 0, "NULL": 0}
            )
            if step_fail:
                failures.append(step_fail)
                if step_undecided:
                    result["v4_counts"]["UNDECIDED"] += 1
                    op_slot["UNDECIDED"] += 1
                else:
                    result["v4_counts"]["NULL"] += 1
                    op_slot["NULL"] += 1
            else:
                passed += 1
                if has_real_change:
                    result["v4_counts"]["FOLLOW"] += 1
                    op_slot["FOLLOW"] += 1
                else:
                    result["v4_counts"]["OMIT"] += 1
                    op_slot["OMIT"] += 1

            if is_goal is not None:
                real_after_f = SimFrame(rec.after, sim_state)
                try:
                    if bool(is_goal(real_after_f)):
                        goal_violations.append(
                            f"GOAL VIOLATION at Attempt {att_id}, Step {rec.step_in_attempt} "
                            f"(seq #{rec.seq_id}): is_goal(frame) returned True on a "
                            f"non-completing historical frame!"
                        )
                except Exception as e:
                    goal_violations.append(f"is_goal raised {type(e).__name__}: {e}")

    # Winning transition verification (Bug #14 fix: require goal_on_pred AND verify physics on pre_level_swap_grid when available)
    if win_rec is not None and is_goal is not None:
        att_state = last_state_by_attempt.get(win_rec.attempt, {})
        win_before_f = SimFrame(win_rec.before, state=att_state)
        try:
            pred_win_out = simulate(win_before_f.copy(), win_rec.action)
            if pred_win_out is None:
                goal_violations.append(
                    f"WINNING TRANSITION VIOLATION (seq #{win_rec.seq_id}, action={win_rec.action}): "
                    f"simulate() returned None on the winning step!"
                )
            else:
                pred_win_f = (
                    pred_win_out if isinstance(pred_win_out, SimFrame)
                    else SimFrame(pred_win_out, win_before_f.state)
                )
                goal_on_pred = bool(is_goal(pred_win_f))
                if not goal_on_pred:
                    goal_violations.append(
                        f"WINNING TRANSITION VIOLATION (seq #{win_rec.seq_id}, action={win_rec.action}): "
                        f"is_goal(simulate(win.before, win.action)) returned False on the actual winning step!"
                    )
                elif not goal_violations and result["ledger_rejection"] is None:
                    result["goal_status"] = "CONFIRMED"
        except Exception as e:
            goal_violations.append(f"Winning step check raised {type(e).__name__}: {e}")

    has_follow = (result["v4_counts"]["FOLLOW"] > 0) or (win_rec is not None)
    # Cross-level transfer fallback when lvl > 1 has only OMIT steps and 0 failures so far (Bug #15)
    if not has_follow and lvl > 1 and not failures and not goal_violations and result["ledger_rejection"] is None:
        for prev_lvl in sorted({r.level for r in wmlib._records if r.level < lvl}, reverse=True):
            if wmlib.transitions(level=prev_lvl) or wmlib.win_transition(prev_lvl) is not None:
                prev_res = evaluate_model(
                    wmlib, falsified_ledger, simulate, is_goal=None,
                    level=prev_lvl, max_unknown_ratio=max_unknown_ratio, project=project,
                )
                if prev_res["ok"] or prev_res["advisory_ok"]:
                    prev_res["report"] = (
                        f"[check_sim] Level {lvl} has 0 FOLLOW steps ({total} OMIT); "
                        f"transferred certification from Level {prev_lvl}:\n" + prev_res["report"]
                    )
                    return prev_res
                break

    if not has_follow:
        failures.append(
            "VACUOUS MODEL VIOLATION (Axiom 4.1 MBEL): FOLLOW=0 — simulator "
            "has not positively explained any active transition on this level."
        )

    if goal_violations or result["ledger_rejection"] is not None:
        result["goal_status"] = "CONTRADICTED"

    accuracy = (passed / total) if total else None
    v4 = result["v4_counts"]
    active_denom = v4["FOLLOW"] + v4["UNDECIDED"] + v4["NULL"]
    active_acc = (v4["FOLLOW"] / active_denom) if active_denom > 0 else accuracy
    coverage = (cov_sum / cov_n) if cov_n else None
    known_acc = (known_ok / known_n) if known_n else None
    ok = (
        (passed == total)
        and has_follow
        and (not goal_violations)
        and (result["ledger_rejection"] is None)
    )
    advisory_ok = (
        not ok
        and has_follow
        and result["ledger_rejection"] is None
        and not goal_violations
        and accuracy is not None
        and accuracy >= ADVISORY_ACC
        and active_acc is not None
        and active_acc >= ADVISORY_ACC
        and (coverage is None or coverage >= ADVISORY_COV)
    )
    op_stats, conf_ops, unconf_ops = _finalize_operator_stats(
        result["operator_stats"], win_rec=win_rec, goal_ok=(result["goal_status"] == "CONFIRMED" or is_goal is None)
    )
    result.update({
        "ok": ok, "advisory_ok": advisory_ok,
        "simulation_accuracy": accuracy, "active_accuracy": active_acc,
        "prediction_coverage": coverage,
        "known_cell_accuracy": known_acc,
        "operator_stats": op_stats,
        "confirmed_operators": conf_ops,
        "unconfirmed_noop_operators": unconf_ops,
        "passed": passed, "total": total,
        "failures": failures[:8], "goal_violations": goal_violations[:5],
        "first_divergence": first_div,
        "per_level": {lvl: {"accuracy": accuracy, "passed": passed, "total": total,
                            "first_divergence": first_div}},
    })
    lines = [
        f"[check_sim] Level {lvl}: {passed}/{total} in-level transitions matched"
        + (f" (accuracy={accuracy:.2f}, active_acc={active_acc:.2f})" if accuracy is not None else "")
        + (f", coverage={coverage:.2f}" if coverage is not None else "")
        + f" | V4: FOLLOW={v4['FOLLOW']} OMIT={v4['OMIT']} U={v4['UNDECIDED']} NULL={v4['NULL']}"
        + f" | goal={result['goal_status']}."
    ]
    if failures:
        lines.append("PHYSICS MISMATCHES:")
        lines.extend("  - " + f for f in failures[:8])
    if goal_violations:
        lines.append("GOAL INVARIANT VIOLATIONS:")
        lines.extend("  - " + g for g in goal_violations[:5])
    if ok:
        lines.append("VERDICT: PASS — simulator and goal are 100% consistent with observed history.")
    elif advisory_ok:
        lines.append(
            f"VERDICT: ADVISORY-OK (accuracy={accuracy:.2f}, active_acc={active_acc:.2f} >= {ADVISORY_ACC:.2f}) — "
            f"search() is allowed; act_checked will guard every step and void the plan "
            f"at the first divergence. Fix the mismatches your planned route touches."
        )
    else:
        lines.append(
            "VERDICT: FAIL — fix the reported mismatches (see first divergence above) "
            "before search() is allowed."
        )
    result["report"] = "\n".join(lines)
    return result


def parse_mouse(action_str):
    """Parse 'MOUSE(row=15, col=20)' -> (15, 20); None for other actions."""
    s = str(action_str).strip()
    if not s.upper().startswith("MOUSE"):
        return None
    import re
    m = re.search(r"row\s*=\s*(-?\d+).*?col\s*=\s*(-?\d+)", s, flags=re.S)
    return (int(m.group(1)), int(m.group(2))) if m else None


def bind_world_model_api(wmlib, falsified_ledger, step_single_action_fn,
                         get_current_grid_fn, get_current_level_fn,
                         notify_hypothesis_falsified_fn):
    """Bind the model-facing CPSA primitives to a live environment stepper."""

    def _step_and_record(act_name):
        prev_grid = get_current_grid_fn().copy()
        lvl_before = get_current_level_fn()
        step_res = step_single_action_fn(act_name)
        if isinstance(step_res, TransitionRecord) or hasattr(step_res, "changed_coords"):
            return step_res
        after_grid, anim_frames, lvl_done, is_dead = step_res
        return wmlib.record_transition(
            action=act_name, before=prev_grid, after=after_grid,
            anim_frames=anim_frames, level_completed=lvl_done,
            game_over=is_dead, level_override=lvl_before,
        )

    def act_expect(actions, expect=None, expect_color_moved=None,
                   max_changed_cells=256, stop_on_hidden_animation=True):
        """Mode 1: execute actions one by one; after EACH step check the
        expectation; on violation abort the remaining tail immediately
        (0 wasted actions) and report the exact diff_text counterexample."""
        action_list = [actions] if isinstance(actions, str) else list(actions)
        if expect is None and expect_color_moved is not None:
            expect = expect_moved(resolve_color(expect_color_moved))
        executed = []
        step_reports = []

        def _finish(status, idx, v4_verdict="UNDECIDED"):
            out = {
                "status": status,
                "verdict_v4": v4_verdict,
                "executed": list(executed),
                "remaining": action_list[idx + 1:],
                "report": "\n".join(step_reports),
            }
            print("\n".join(step_reports))
            return out

        for idx, act_name in enumerate(action_list):
            rec = _step_and_record(act_name)
            executed.append(act_name)
            if rec.level_completed:
                step_reports.append(f"Step {idx + 1} ({act_name}): LEVEL {rec.level} COMPLETED!")
                return _finish("LEVEL_COMPLETED", idx, "FOLLOW")
            if rec.game_over:
                msg = (
                    f"Step {idx + 1} ({act_name}): GAME_OVER triggered! Changed cells: "
                    f"{len(rec.changed_coords)}. Remaining steps aborted."
                )
                step_reports.append(msg)
                notify_hypothesis_falsified_fn(msg)
                return _finish("GAME_OVER", idx, "NULL")

            hud_mask = wmlib.compute_hud_mask(rec.level)
            non_hud = [(r, c, o, n) for (r, c, o, n) in rec.changed_coords if not hud_mask[r, c]]

            if stop_on_hidden_animation and len(non_hud) == 0 and rec.num_anim_frames > 1:
                msg = (
                    f"Step {idx + 1} ({act_name}): HALTED ON HIDDEN ANIMATION! Final grid had "
                    f"0 non-HUD changes but {rec.num_anim_frames} animation frames occurred. "
                    f"Inspect `last_animation_frames` — a transient effect (beam/bounce/flash) "
                    f"carries information your model is missing."
                )
                step_reports.append(msg)
                notify_hypothesis_falsified_fn(msg)
                return _finish("HIDDEN_ANIMATION_STOP", idx, "UNDECIDED")

            if len(non_hud) == 0 and len(action_list) > 1:
                msg = (
                    f"Step {idx + 1} ({act_name}): HALTED ON UNEXPECTED NO-OP! 0 non-HUD cells "
                    f"changed. Remaining {len(action_list) - idx - 1} steps aborted."
                )
                step_reports.append(msg)
                notify_hypothesis_falsified_fn(msg)
                return _finish("EXPECTATION_MISMATCH", idx, "OMIT")

            if len(non_hud) > max_changed_cells:
                msg = (
                    f"Step {idx + 1} ({act_name}): HALTED ON MASSIVE BOARD CHANGE! "
                    f"{len(non_hud)} cells changed (max_changed_cells={max_changed_cells})."
                )
                step_reports.append(msg)
                return _finish("MASSIVE_CHANGE_STOP", idx, "UNDECIDED")

            if expect is not None:
                prev_f = SimFrame(rec.before)
                curr_f = SimFrame(rec.after)
                res = expect(prev_f, curr_f, act_name, non_hud)
                ok, reason = res if isinstance(res, tuple) else (bool(res), "")
                if not ok:
                    compact = diff_text(rec.before, rec.after)
                    msg = (
                        f"Step {idx + 1} ({act_name}): EXPECTATION MISMATCH! {reason}\n"
                        f"Actual transition:\n{compact}\n"
                        f"Aborted remaining {len(action_list) - idx - 1} planned steps."
                    )
                    step_reports.append(msg)
                    notify_hypothesis_falsified_fn(msg)
                    return _finish("EXPECTATION_MISMATCH", idx, "NULL")
                step_reports.append(
                    f"Step {idx + 1} ({act_name}): OK ({reason or str(len(non_hud)) + ' cells changed'})"
                )
            else:
                step_reports.append(f"Step {idx + 1} ({act_name}): {len(non_hud)} non-HUD cells changed.")
        print("\n".join(step_reports))
        return {
            "status": "COMPLETED",
            "verdict_v4": "FOLLOW" if (expect is not None or len(non_hud) > 0) else "OMIT",
            "executed": executed,
            "remaining": [],
            "report": "\n".join(step_reports),
        }

    def check_sim(simulate, is_goal=None, level=None, only_active_attempt=False,
                  border=0, max_unknown_ratio=MAX_UNKNOWN_RATIO, project=None):
        """Mode 2: the Deterministic V4 Judge. Zero environment cost.
        Pass `project=project` to verify a Boundary Projection (quotient) model."""
        res = evaluate_model(
            wmlib, falsified_ledger, simulate, is_goal,
            level=level, max_unknown_ratio=max_unknown_ratio, project=project,
        )
        print(res["report"])
        return res

    def get_current_sim_frame(simulate=None):
        cur_grid = get_current_grid_fn().copy()
        if simulate is None:
            return SimFrame(cur_grid, state={})
        recs = wmlib.transitions(
            only_active_attempt=True, include_game_over=False,
            include_level_completed=False,
        )
        if not recs:
            return SimFrame(cur_grid, state={})
        f = SimFrame(recs[0].before, state={})
        for r in recs:
            f.grid = r.before.astype(np.int16).copy()
            try:
                out = simulate(f.copy(), r.action)
                if isinstance(out, SimFrame):
                    f.state = copy_state(out.state)
            except Exception:
                break
        f.grid = cur_grid.astype(np.int16).copy()
        return f

    def get_current_projected_state(project, simulate=None):
        cur_grid = get_current_grid_fn().copy()
        obs_now = project(SimFrame(cur_grid))
        if simulate is None:
            return obs_now
        recs = wmlib.transitions(
            only_active_attempt=True, include_game_over=False,
            include_level_completed=False,
        )
        if not recs:
            return obs_now
        st = project(SimFrame(recs[0].before))
        for r in recs:
            obs_b = project(SimFrame(r.before))
            if _canonicalize_state(_observable_projection(st)) != _canonicalize_state(_observable_projection(obs_b)):
                st = obs_b
            try:
                nxt = simulate(copy_state(st), r.action)
                if nxt is not None:
                    st = nxt
            except Exception:
                return obs_now
        if _canonicalize_state(_observable_projection(st)) == _canonicalize_state(_observable_projection(obs_now)):
            return st
        return obs_now

    def explore_frontier(frame_or_grid, simulate,
                         actions=("UP", "DOWN", "LEFT", "RIGHT"),
                         max_depth=25, max_states=5000, project=None):
        """Zero-cost Reachable Interaction Frontier (LAF §5.2 Loop Closure / Use).
        Expands verified `simulate` from `frame_or_grid` and returns shortest action
        sequences that trigger novel object/color changes or state-key transitions
        when the final win condition is unknown or unreachable."""
        lvl = get_current_level_fn()
        hud_mask = wmlib.compute_hud_mask(lvl)
        active_mask = ~hud_mask
        action_list = list(actions)

        if isinstance(frame_or_grid, SimFrame):
            start_f = frame_or_grid.copy()
        else:
            start_f = get_current_sim_frame(simulate if project is None else None)
            start_f.grid = _extract_grid(frame_or_grid).astype(np.int16, copy=True)

        frontier = []
        seen_events = set()

        if project is not None:
            start_st = project(start_f.copy()) if (isinstance(frame_or_grid, SimFrame) and frame_or_grid.state) else get_current_projected_state(project, simulate)
            visited = {_canonicalize_state(start_st)}
            queue = deque([(start_st, [])])
            expanded = 0
            while queue and expanded < max_states and len(frontier) < 8:
                cur_st, path = queue.popleft()
                if len(path) >= max_depth:
                    continue
                expanded += 1
                for act in action_list:
                    try:
                        nxt_st = simulate(copy_state(cur_st), act)
                    except Exception:
                        continue
                    if nxt_st is None or _is_dead_state(nxt_st):
                        continue
                    k = _canonicalize_state(nxt_st)
                    if k in visited:
                        continue
                    visited.add(k)
                    new_path = path + [act]
                    if isinstance(nxt_st, dict) and isinstance(start_st, dict):
                        for key_name, val in nxt_st.items():
                            if key_name in start_st and _canonicalize_state(val) != _canonicalize_state(start_st[key_name]):
                                ev_key = f"key:{key_name}"
                                if ev_key not in seen_events:
                                    seen_events.add(ev_key)
                                    frontier.append({
                                        "event": f"state[{key_name!r}] changes {start_st[key_name]!r} -> {val!r}",
                                        "plan": new_path,
                                        "depth": len(new_path),
                                    })
                    elif len(new_path) in (1, max(1, max_depth // 2)):
                        ev_key = f"depth:{len(new_path)}"
                        if ev_key not in seen_events:
                            seen_events.add(ev_key)
                            frontier.append({
                                "event": f"projected state {_observable_projection(nxt_st)!r}",
                                "plan": new_path,
                                "depth": len(new_path),
                            })
                    queue.append((nxt_st, new_path))
            wmlib.last_search_frontier = frontier
            return frontier

        u_vals, u_cnts = np.unique(start_f.grid[active_mask], return_counts=True)
        bg_val = int(u_vals[np.argmax(u_cnts)]) if len(u_vals) else 0
        init_counts = {int(c): int(cnt) for c, cnt in zip(u_vals, u_cnts) if int(c) != bg_val}
        static_locs = {
            int(c): set(zip(*np.nonzero((start_f.grid == int(c)) & active_mask)))
            for c in init_counts
        }

        def _state_key(f):
            return (f.grid[active_mask].tobytes(), _canonicalize_state(f.state))

        visited = {_state_key(start_f)}
        queue = deque([(start_f, [])])
        expanded = 0
        avatar_colors = set()

        while queue and expanded < max_states and len(frontier) < 8:
            cur_f, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            expanded += 1
            for act in action_list:
                try:
                    nxt_out = simulate(cur_f.copy(), act)
                except Exception:
                    continue
                if nxt_out is None:
                    continue
                nxt_f = nxt_out if isinstance(nxt_out, SimFrame) else SimFrame(nxt_out, cur_f.state)
                if _is_dead_state(nxt_f):
                    continue
                k = _state_key(nxt_f)
                if k in visited:
                    continue
                visited.add(k)
                new_path = path + [act]

                diff = (nxt_f.grid != cur_f.grid) & active_mask
                if np.any(diff):
                    changed_colors = set(int(x) for x in nxt_f.grid[diff]) | set(int(x) for x in cur_f.grid[diff])
                    changed_colors.discard(bg_val)
                    if len(new_path) == 1:
                        avatar_colors |= changed_colors
                    for c_val in sorted(changed_colors):
                        if c_val not in avatar_colors:
                            ev_key = f"modify_color:{c_val}"
                            if ev_key not in seen_events:
                                seen_events.add(ev_key)
                                frontier.append({
                                    "event": f"modifies object color {c_val} ('{color_char(c_val)}')",
                                    "plan": new_path,
                                    "depth": len(new_path),
                                })
                    for ac in avatar_colors:
                        av_rs, av_cs = np.nonzero((nxt_f.grid == ac) & active_mask)
                        for r_a, c_a in zip(av_rs, av_cs):
                            for sc, sc_pts in static_locs.items():
                                if sc in avatar_colors:
                                    continue
                                ev_key = f"touch_color:{sc}"
                                if ev_key in seen_events:
                                    continue
                                if (
                                    (int(r_a) - 1, int(c_a)) in sc_pts
                                    or (int(r_a) + 1, int(c_a)) in sc_pts
                                    or (int(r_a), int(c_a) - 1) in sc_pts
                                    or (int(r_a), int(c_a) + 1) in sc_pts
                                ):
                                    seen_events.add(ev_key)
                                    frontier.append({
                                        "event": f"reaches adjacency to object color {sc} ('{color_char(sc)}') at ({int(r_a)},{int(c_a)})",
                                        "plan": new_path,
                                        "depth": len(new_path),
                                    })
                queue.append((nxt_f, new_path))

        wmlib.last_search_frontier = frontier
        return frontier

    def search(frame_or_grid, simulate, is_goal,
               actions=("UP", "DOWN", "LEFT", "RIGHT"),
               max_depth=50, max_states=25000, heuristic=None,
               allow_unchecked_goal=False, project=None,
               strict_operator_confirmation=False):
        """Mode 3: BFS (heuristic=None) or A* over SimFrame or projected state
        pi_K(frame) when `project` is supplied."""
        lvl = get_current_level_fn()
        if not allow_unchecked_goal:
            verdict = evaluate_model(
                wmlib, falsified_ledger, simulate, is_goal, level=lvl, project=project
            )
            if verdict["ledger_rejection"]:
                raise RuntimeError(verdict["ledger_rejection"])
            if verdict["ok"]:
                tier = "VERIFIED"
            elif verdict["advisory_ok"]:
                tier = "ADVISORY"
                print(
                    f"[search] MODEL TIER: ADVISORY (accuracy="
                    f"{verdict['simulation_accuracy']:.2f}). The plan will be guarded "
                    f"step-by-step by act_checked; avoid routes through known-mismatch "
                    f"situations. First divergence: "
                    f"{(verdict['first_divergence'] or {}).get('action', 'n/a')}"
                )
            else:
                raise RuntimeError(
                    "REJECTED BY JUDGE: model is not consistent enough to plan.\n"
                    + verdict["report"]
                )
        else:
            tier = "UNCHECKED"

        if isinstance(frame_or_grid, SimFrame):
            start_f = frame_or_grid.copy()
        else:
            start_f = get_current_sim_frame(simulate if project is None else None)
            start_f.grid = _extract_grid(frame_or_grid).astype(np.int16, copy=True)

        action_list = list(actions)
        if not allow_unchecked_goal and strict_operator_confirmation:
            unconf_noop = set(verdict.get("unconfirmed_noop_operators") or [])
            conf_ops = set(verdict.get("confirmed_operators") or [])
            dirs_set = {"UP", "DOWN", "LEFT", "RIGHT"}
            any_dir_confirmed = bool(conf_ops & dirs_set)
            filtered_actions = []
            for a in action_list:
                if a in unconf_noop and not (a in dirs_set and any_dir_confirmed):
                    continue
                filtered_actions.append(a)
            action_list = filtered_actions

        if project is not None:
            if isinstance(frame_or_grid, SimFrame) and frame_or_grid.state:
                start_st = project(start_f.copy())
            else:
                start_st = get_current_projected_state(project, simulate)
            if is_goal(copy_state(start_st)):
                print("[search] Warning: is_goal(project(start_frame)) is ALREADY True!")
                return []
            visited = {_canonicalize_state(start_st)}
            expanded = 0
            if heuristic is None:
                queue = deque([(start_st, [])])
                while queue and expanded < max_states:
                    cur_st, path = queue.popleft()
                    if len(path) >= max_depth:
                        continue
                    expanded += 1
                    for act in action_list:
                        try:
                            nxt_st = simulate(copy_state(cur_st), act)
                        except Exception:
                            continue
                        if nxt_st is None or _is_dead_state(nxt_st):
                            continue
                        k = _canonicalize_state(nxt_st)
                        if k in visited:
                            continue
                        visited.add(k)
                        new_path = path + [act]
                        try:
                            if is_goal(copy_state(nxt_st)):
                                print(
                                    f"[search/{tier}/PROJECTED] Found shortest plan of length "
                                    f"{len(new_path)} (expanded {expanded}): {new_path}"
                                )
                                return new_path
                        except Exception:
                            pass
                        queue.append((nxt_st, new_path))
            else:
                counter = 0
                heap = [(float(heuristic(copy_state(start_st))), 0, start_st, [])]
                while heap and expanded < max_states:
                    _, _, cur_st, path = heapq.heappop(heap)
                    if len(path) >= max_depth:
                        continue
                    expanded += 1
                    for act in action_list:
                        try:
                            nxt_st = simulate(copy_state(cur_st), act)
                        except Exception:
                            continue
                        if nxt_st is None or _is_dead_state(nxt_st):
                            continue
                        k = _canonicalize_state(nxt_st)
                        if k in visited:
                            continue
                        visited.add(k)
                        new_path = path + [act]
                        try:
                            if is_goal(copy_state(nxt_st)):
                                print(
                                    f"[search A*/{tier}/PROJECTED] Found plan of length "
                                    f"{len(new_path)} (expanded {expanded}): {new_path}"
                                )
                                return new_path
                        except Exception:
                            pass
                        counter += 1
                        heapq.heappush(
                            heap,
                            (len(new_path) + float(heuristic(copy_state(nxt_st))), counter, nxt_st, new_path),
                        )
            frontier = explore_frontier(
                frame_or_grid, simulate, actions=action_list,
                max_depth=min(max_depth, 25), max_states=min(max_states, 2000), project=project,
            )
            f_msg = ""
            if frontier:
                f_msg = "\nReachable Interaction Frontier (0 env cost subgoals):\n" + "\n".join(
                    f"  * depth {item['depth']}: {item['event']} -> plan={item['plan']}"
                    for item in frontier[:5]
                )
            print(
                f"[search/{tier}/PROJECTED] No plan found within max_depth={max_depth}, "
                f"max_states={max_states} (expanded={expanded}, unique={len(visited)})."
                + f_msg
            )
            return None

        hud_mask = wmlib.compute_hud_mask(lvl)
        active_mask = ~hud_mask

        def _state_key(f):
            return (f.grid[active_mask].tobytes(), _canonicalize_state(f.state))

        if is_goal(start_f):
            print("[search] Warning: is_goal(start_frame) is ALREADY True on the current board!")
            return []
        visited = {_state_key(start_f)}
        expanded = 0

        if heuristic is None:
            queue = deque([(start_f, [])])
            while queue and expanded < max_states:
                cur_f, path = queue.popleft()
                if len(path) >= max_depth:
                    continue
                expanded += 1
                for act in action_list:
                    try:
                        nxt_out = simulate(cur_f.copy(), act)
                    except Exception:
                        continue
                    if nxt_out is None:
                        continue
                    nxt_f = nxt_out if isinstance(nxt_out, SimFrame) else SimFrame(nxt_out, cur_f.state)
                    if _is_dead_state(nxt_f):
                        continue
                    k = _state_key(nxt_f)
                    if k in visited:
                        continue
                    visited.add(k)
                    new_path = path + [act]
                    try:
                        if is_goal(nxt_f):
                            print(
                                f"[search/{tier}] Found shortest plan of length {len(new_path)} "
                                f"(expanded {expanded}): {new_path}"
                            )
                            return new_path
                    except Exception:
                        pass
                    queue.append((nxt_f, new_path))
        else:
            counter = 0
            heap = [(float(heuristic(start_f)), 0, start_f, [])]
            while heap and expanded < max_states:
                _, _, cur_f, path = heapq.heappop(heap)
                if len(path) >= max_depth:
                    continue
                expanded += 1
                for act in action_list:
                    try:
                        nxt_out = simulate(cur_f.copy(), act)
                    except Exception:
                        continue
                    if nxt_out is None:
                        continue
                    nxt_f = nxt_out if isinstance(nxt_out, SimFrame) else SimFrame(nxt_out, cur_f.state)
                    if _is_dead_state(nxt_f):
                        continue
                    k = _state_key(nxt_f)
                    if k in visited:
                        continue
                    visited.add(k)
                    new_path = path + [act]
                    try:
                        if is_goal(nxt_f):
                            print(
                                f"[search A*/{tier}] Found plan of length {len(new_path)} "
                                f"(expanded {expanded}): {new_path}"
                            )
                            return new_path
                    except Exception:
                        pass
                    counter += 1
                    heapq.heappush(
                        heap,
                        (len(new_path) + float(heuristic(nxt_f)), counter, nxt_f, new_path),
                    )

        frontier = explore_frontier(
            frame_or_grid, simulate, actions=action_list,
            max_depth=min(max_depth, 25), max_states=min(max_states, 2000), project=None,
        )
        f_msg = ""
        if frontier:
            f_msg = "\nReachable Interaction Frontier (0 env cost subgoals):\n" + "\n".join(
                f"  * depth {item['depth']}: {item['event']} -> plan={item['plan']}"
                for item in frontier[:5]
            )
        print(
            f"[search/{tier}] No plan found within max_depth={max_depth}, "
            f"max_states={max_states} (expanded={expanded}, unique={len(visited)}). "
            f"A timed-out/exhausted search means the SEARCH WAS TOO BIG or the MODEL IS "
            f"MISSING something (an object, a hidden state variable, an affordance of "
            f"something you assumed was decoration, or an is_goal stub that never returns True) "
            f"— never that the level is impossible."
            + f_msg
        )
        return None

    def act_checked(plan, simulate, is_goal=None, project=None):
        """Mode 3 executor (Kepler commit semantics): predict every step; the
        FIRST divergence voids the remaining plan (0 wasted actions) and
        returns the diff_text counterexample."""
        if not plan:
            msg = "[act_checked] Plan is empty or None; 0 actions executed."
            print(msg)
            return {"status": "EMPTY_PLAN", "executed": [], "remaining": [], "report": msg}
        lvl = get_current_level_fn()
        hud_mask = wmlib.compute_hud_mask(lvl)
        sim_f = get_current_sim_frame(simulate if project is None else None)
        proj_st = get_current_projected_state(project, simulate) if project is not None else None
        executed = []

        for idx, act_name in enumerate(plan):
            prev_grid = get_current_grid_fn().copy()
            sim_f.grid = prev_grid.astype(np.int16).copy()

            if project is not None:
                try:
                    s_before_obs = project(SimFrame(prev_grid))
                    if (
                        proj_st is not None
                        and _canonicalize_state(_observable_projection(proj_st))
                        == _canonicalize_state(_observable_projection(s_before_obs))
                    ):
                        s_in = proj_st
                    else:
                        s_in = s_before_obs
                    s_pred = simulate(copy_state(s_in), act_name)
                except Exception as e:
                    msg = (
                        f"[act_checked] projected simulate() raised {type(e).__name__}: {e} "
                        f"at step {idx + 1} ({act_name}); remaining plan voided."
                    )
                    print(msg)
                    notify_hypothesis_falsified_fn(msg)
                    return {"status": "SIM_ERROR", "executed": executed,
                            "remaining": list(plan[idx:]), "report": msg}

                rec = _step_and_record(act_name)
                executed.append(act_name)
                if rec.level_completed:
                    msg = f"[act_checked] LEVEL {rec.level} COMPLETED at step {idx + 1}/{len(plan)} ({act_name})!"
                    print(msg)
                    return {"status": "LEVEL_COMPLETED", "executed": executed,
                            "remaining": list(plan[idx + 1:]), "report": msg}
                if rec.game_over:
                    msg = (
                        f"[act_checked] GAME_OVER at step {idx + 1}/{len(plan)} ({act_name})! "
                        f"Your simulator did not predict this hazard."
                    )
                    print(msg)
                    notify_hypothesis_falsified_fn(msg)
                    return {"status": "GAME_OVER", "executed": executed,
                            "remaining": list(plan[idx + 1:]), "report": msg}

                proj_st = copy_state(s_pred)
                s_actual = project(SimFrame(rec.after))
                if _canonicalize_state(_observable_projection(s_pred)) != _canonicalize_state(_observable_projection(s_actual)):
                    compact = diff_text(rec.before, rec.after)
                    msg = (
                        f"[act_checked] PROJECTED SIMULATOR DIVERGENCE at step {idx + 1}/{len(plan)} "
                        f"({act_name})! Aborted remaining {len(plan) - idx - 1} steps.\n"
                        f"  predicted state: {s_pred!r}\n"
                        f"  observed state:  {s_actual!r}\n"
                        f"  board diff:\n{compact}"
                    )
                    print(msg)
                    notify_hypothesis_falsified_fn(msg)
                    return {"status": "SIM_DIVERGENCE", "executed": executed,
                            "remaining": list(plan[idx + 1:]), "report": msg}

                if is_goal is not None:
                    try:
                        goal_hit = bool(is_goal(copy_state(s_actual)))
                    except Exception:
                        goal_hit = False
                    if goal_hit:
                        falsified_ledger.record_false_positive_goal(
                            level=lvl, grid=rec.after, state=s_actual,
                            seq_id=rec.seq_id, is_goal_fn=is_goal,
                        )
                        msg = (
                            f"[act_checked] GOAL HYPOTHESIS FALSIFIED at step {idx + 1}/{len(plan)} "
                            f"({act_name})! Transition #{rec.seq_id} blacklisted in FalsifiedGoalLedger."
                        )
                        print(msg)
                        notify_hypothesis_falsified_fn(msg)
                        return {"status": "GOAL_FALSIFIED", "executed": executed,
                                "remaining": list(plan[idx + 1:]), "report": msg}
                continue

            try:
                pred_out = simulate(sim_f.copy(), act_name)
                pred_err = None
            except Exception as e:
                pred_out = None
                pred_err = f"{type(e).__name__}: {e}"
            if pred_out is None:
                msg = (
                    f"[act_checked] simulate() raised {pred_err} at step {idx + 1} "
                    f"({act_name}); remaining plan voided."
                )
                print(msg)
                notify_hypothesis_falsified_fn(msg)
                return {"status": "SIM_ERROR", "executed": executed,
                        "remaining": list(plan[idx:]), "report": msg}
            pred_f = pred_out if isinstance(pred_out, SimFrame) else SimFrame(pred_out, sim_f.state)
            sim_f.state = copy_state(pred_f.state)

            rec = _step_and_record(act_name)
            executed.append(act_name)

            if rec.level_completed:
                msg = f"[act_checked] LEVEL {rec.level} COMPLETED at step {idx + 1}/{len(plan)} ({act_name})!"
                print(msg)
                return {"status": "LEVEL_COMPLETED", "executed": executed,
                        "remaining": list(plan[idx + 1:]), "report": msg}
            if rec.game_over:
                msg = (
                    f"[act_checked] GAME_OVER at step {idx + 1}/{len(plan)} ({act_name})! "
                    f"Your simulator did not predict this hazard. Encode what killed you "
                    f"(save_invariants(hazard_or_falsified=...)) and update simulate()/is_goal()."
                )
                print(msg)
                notify_hypothesis_falsified_fn(msg)
                return {"status": "GAME_OVER", "executed": executed,
                        "remaining": list(plan[idx + 1:]), "report": msg}

            check_mask = (~hud_mask) & (pred_f.grid != UNKNOWN_CELL)
            mismatches = check_mask & (pred_f.grid != rec.after.astype(np.int16))
            if np.any(mismatches):
                actual_masked = rec.after.astype(np.int16).copy()
                pred_masked = pred_f.grid.copy()
                actual_masked[~check_mask] = 0
                pred_masked[~check_mask] = 0
                compact = diff_text(actual_masked, pred_masked)
                msg = (
                    f"[act_checked] SIMULATOR DIVERGENCE at step {idx + 1}/{len(plan)} "
                    f"({act_name})! Aborted remaining {len(plan) - idx - 1} steps.\n"
                    f"predicted-vs-actual (masked):\n{compact}\n"
                    f"This transition is now in wmlib — run check_sim(simulate, is_goal) "
                    f"and fix the rule (or the OBJECT vocabulary) it falsified."
                )
                print(msg)
                notify_hypothesis_falsified_fn(msg)
                return {"status": "SIM_DIVERGENCE", "executed": executed,
                        "remaining": list(plan[idx + 1:]), "report": msg}

            if is_goal is not None:
                actual_f = SimFrame(rec.after, sim_f.state)
                try:
                    goal_hit = bool(is_goal(actual_f))
                except Exception:
                    goal_hit = False
                if goal_hit:
                    falsified_ledger.record_false_positive_goal(
                        level=lvl, grid=rec.after, state=sim_f.state,
                        seq_id=rec.seq_id, is_goal_fn=is_goal,
                    )
                    msg = (
                        f"[act_checked] GOAL HYPOTHESIS FALSIFIED at step {idx + 1}/{len(plan)} "
                        f"({act_name})! Board reached a state where is_goal(frame)==True but the "
                        f"level DID NOT complete. Transition #{rec.seq_id} blacklisted in "
                        f"FalsifiedGoalLedger — formulate a new is_goal(frame) (the objective "
                        f"is an inference: study what changes at REAL level ends)."
                    )
                    print(msg)
                    notify_hypothesis_falsified_fn(msg)
                    return {"status": "GOAL_FALSIFIED", "executed": executed,
                            "remaining": list(plan[idx + 1:]), "report": msg}

        msg = (
            f"[act_checked] Executed all {len(executed)} steps matching the simulator, "
            f"but the level did not complete. Re-examine is_goal / the route."
        )
        print(msg)
        return {"status": "COMPLETED_NO_WIN", "executed": executed, "remaining": [], "report": msg}

    return {
        "SimFrame": SimFrame,
        "ARC_COLOR_CHARS": ARC_COLOR_CHARS,
        "resolve_color": resolve_color,
        "color_char": color_char,
        "expect_moved": expect_moved,
        "expect_region_changed": expect_region_changed,
        "expect_diff_count": expect_diff_count,
        "expect_cells": expect_cells,
        "act_expect": act_expect,
        "check_sim": check_sim,
        "evaluate_model": evaluate_model,
        "get_current_sim_frame": get_current_sim_frame,
        "get_current_projected_state": get_current_projected_state,
        "search": search,
        "explore_frontier": explore_frontier,
        "act_checked": act_checked,
        "diff_text": diff_text,
        "cells": cells,
        "bbox": bbox,
        "colors": colors,
        "show_window": show_window,
        "candidate_clicks": candidate_clicks,
        "parse_mouse": parse_mouse,
    }
