"""cpsa.wmlib — transition journal, color resolution, HUD detector, compact diff.

Self-contained: stdlib + numpy only. No project imports. No __future__.
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

GRID_H = 64
GRID_W = 64
DIFF_TEXT_MAX_LINES = 12

# Exact match with inference.utils.grid_utils.ARC_COLOR_CHARS
ARC_COLOR_CHARS = "WwgGcBMPRbSYOrNp"
_CHAR_TO_COLOR = {ch: idx for idx, ch in enumerate(ARC_COLOR_CHARS)}


def resolve_color(color):
    """Accept either an integer color ID (0..15) or an ARC color character
    ('W'=0, 'w'=1, 'g'=2, 'G'=3, 'c'=4, 'B'=5, 'M'=6, 'P'=7, 'R'=8, 'b'=9,
    'S'=10, 'Y'=11, 'O'=12, 'r'=13, 'N'=14, 'p'=15)."""
    if isinstance(color, str):
        s = color.strip()
        if s in _CHAR_TO_COLOR:
            return _CHAR_TO_COLOR[s]
        return int(s)
    return int(color)


def color_char(color_id):
    """Convert integer color ID (0..15) to its ARC character."""
    idx = max(0, min(15, int(color_id)))
    return ARC_COLOR_CHARS[idx]


def diff_text(prev, nxt):
    """Compact rectangular run-length description of changed cells (Tycho-style).

    Groups changes by (old_color -> new_color) pair, run-length-encodes columns
    per row, and merges consecutive rows with identical column specs.
    Output is capped at DIFF_TEXT_MAX_LINES region lines.
    """
    a = np.asarray(prev, dtype=np.int16)
    b = np.asarray(nxt, dtype=np.int16)
    diff_mask = a != b
    if not np.any(diff_mask):
        return "no cells changed"
    coords = np.argwhere(diff_mask)
    rs, cs = coords[:, 0], coords[:, 1]
    bbox_str = (
        f"rows {int(rs.min())}-{int(rs.max())}, "
        f"cols {int(cs.min())}-{int(cs.max())}"
    )

    def _runs(nums):
        ordered = sorted(set(nums))
        runs = []
        start = end = ordered[0]
        for n in ordered[1:]:
            if n == end + 1:
                end = n
            else:
                runs.append(f"{start}-{end}" if start != end else str(start))
                start = end = n
        runs.append(f"{start}-{end}" if start != end else str(start))
        return ",".join(runs)

    by_trans = defaultdict(lambda: defaultdict(list))
    for r, c in coords:
        by_trans[(int(a[r, c]), int(b[r, c]))][int(r)].append(int(c))

    regions = []
    for (o, n), rows in by_trans.items():
        rowspec = [(r, _runs(rows[r])) for r in sorted(rows)]
        i = 0
        while i < len(rowspec):
            r0, spec = rowspec[i]
            j = i
            while (
                j + 1 < len(rowspec)
                and rowspec[j + 1][1] == spec
                and rowspec[j + 1][0] == rowspec[j][0] + 1
            ):
                j += 1
            rng = f"rows {r0}-{rowspec[j][0]}" if j > i else f"row {r0}"
            n_cells = sum(
                len(set(rows[r]))
                for r in range(rowspec[i][0], rowspec[j][0] + 1)
                if r in rows
            )
            regions.append((n_cells, f"  {o}->{n}  {rng}, cols {spec}"))
            i = j + 1
    regions.sort(key=lambda item: -item[0])
    lines = [text for _, text in regions[:DIFF_TEXT_MAX_LINES]]
    if len(regions) > DIFF_TEXT_MAX_LINES:
        extra = len(regions) - DIFF_TEXT_MAX_LINES
        lines.append(f"  (+{extra} more regions)")
    return f"{len(coords)} cells changed; region [{bbox_str}]:\n" + "\n".join(lines)


@dataclass
class TransitionRecord:
    seq_id: int
    level: int
    attempt: int
    step_in_attempt: int
    action: str
    before: np.ndarray
    after: np.ndarray
    changed_coords: list
    num_anim_frames: int
    anim_frames: list = field(default_factory=list)
    level_completed: bool = False
    game_over: bool = False
    automatic: bool = False

    @property
    def is_noop(self):
        return len(self.changed_coords) == 0 and self.num_anim_frames <= 1

    @property
    def has_hidden_animation(self):
        return len(self.changed_coords) == 0 and self.num_anim_frames > 1

    @property
    def pre_level_swap_grid(self):
        """Terminal board of the finished level (frame before the level swap).

        In solver.py, collapse_frames prepends `before` at index 0 and places
        the new level's starting board at index -1. When there is no intermediate
        animation frame (len == 2 and anim_frames[0] == before), returns None
        so the Judge never mistakes `before` for the post-move terminal frame.
        """
        if not self.level_completed or len(self.anim_frames) < 2:
            return None
        cand = self.anim_frames[-2]
        if len(self.anim_frames) == 2 and np.array_equal(cand, self.before):
            return None
        return cand


class WorldModelLibrary:
    """Append-only journal of every transition of the game (all levels/attempts)."""

    def __init__(self):
        self._records = []
        self._current_level = 1
        self._current_attempt = 1
        self._step_in_attempt = 0
        self._hud_mask_cache = {}
        self._level_start_frames = {}
        self._level_win_records = {}
        self.last_search_frontier = []

    @property
    def current_level(self):
        return self._current_level

    @property
    def current_attempt(self):
        return self._current_attempt

    @property
    def step_in_attempt(self):
        return self._step_in_attempt

    @property
    def last_transition(self):
        return self._records[-1] if self._records else None

    def on_level_start(self, level, start_grid):
        lvl = int(level)
        self._current_level = lvl
        self._current_attempt = 1
        self._step_in_attempt = 0
        if lvl not in self._level_start_frames:
            self._level_start_frames[lvl] = np.array(start_grid, dtype=np.int8, copy=True)

    def on_attempt_reset(self):
        if self._step_in_attempt == 0 and self._current_attempt > 1:
            return
        self._current_attempt += 1
        self._step_in_attempt = 0

    def level_start_grid(self, level=None):
        lvl = self._current_level if level is None else int(level)
        arr = self._level_start_frames.get(lvl)
        return arr.copy() if arr is not None else None

    def record_transition(self, action, before, after, anim_frames=None,
                          level_completed=False, game_over=False,
                          level_override=None, automatic=False):
        lvl = self._current_level if level_override is None else int(level_override)
        self._step_in_attempt += 1
        b = np.array(before, dtype=np.int8, copy=True)
        a = np.array(after, dtype=np.int8, copy=True)
        ys, xs = np.nonzero(b != a)
        changed = [(int(r), int(c), int(b[r, c]), int(a[r, c])) for r, c in zip(ys, xs)]
        af = [np.array(f, dtype=np.int8, copy=True) for f in (anim_frames or [])]
        rec = TransitionRecord(
            seq_id=len(self._records) + 1,
            level=lvl,
            attempt=self._current_attempt,
            step_in_attempt=self._step_in_attempt,
            action=str(action),
            before=b,
            after=a,
            changed_coords=changed,
            num_anim_frames=max(1, len(af)),
            anim_frames=af,
            level_completed=bool(level_completed),
            game_over=bool(game_over),
            automatic=bool(automatic),
        )
        self._records.append(rec)
        if rec.level_completed:
            self._level_win_records[lvl] = rec
        self._hud_mask_cache.pop(lvl, None)
        return rec

    def transitions(self, level=None, attempt=None, action=None,
                    only_active_attempt=False, include_game_over=True,
                    include_level_completed=True, include_reset=False):
        target_level = self._current_level if level is None else int(level)
        out = []
        for r in self._records:
            if target_level != -1 and r.level != target_level:
                continue
            if not include_reset and (r.automatic or str(r.action).strip().upper().startswith("RESET")):
                continue
            if only_active_attempt and r.attempt != self._current_attempt:
                continue
            if attempt is not None and r.attempt != int(attempt):
                continue
            if action is not None and r.action.upper() != str(action).upper():
                continue
            if not include_game_over and r.game_over:
                continue
            if not include_level_completed and r.level_completed:
                continue
            out.append(r)
        return out

    def win_transition(self, level=None):
        lvl = self._current_level if level is None else int(level)
        return self._level_win_records.get(lvl)

    def compute_hud_mask(self, level=None):
        """Boolean 64x64 mask of edge lines (rows/cols 0,1,62,63) that tick
        regularly (1..4 pixels per step) = timer/budget HUD, not gameplay.

        CPSA 2.1 upgrades:
          1. Inherits the previous level's detected HUD mask when the target level
             has fewer than 2 active transitions (Turn 0 / Turn 1 on Level 2+).
          2. Detects HUD after >= 2 steps even when the same action is repeated
             (e.g. ["DOWN", "DOWN"]) when the edge line changes by 1px at adjacent
             positions.
          3. Requires the edge line to tick on >= 70% of active transitions so a
             1x1/2x2 player briefly walking along row 1 is never misclassified as HUD.
        """
        lvl = self._current_level if level is None else int(level)
        if lvl in self._hud_mask_cache:
            return self._hud_mask_cache[lvl]
        mask = np.zeros((GRID_H, GRID_W), dtype=bool)
        recs = [
            r for r in self._records
            if r.level == lvl and not r.level_completed and not r.game_over
            and not r.automatic and not str(r.action).strip().upper().startswith("RESET")
        ]
        active_recs = [r for r in recs if not np.array_equal(r.before, r.after)]
        if len(active_recs) < 2:
            if lvl > 1:
                for prev_lvl in sorted({r.level for r in self._records if r.level < lvl}, reverse=True):
                    prev_mask = self.compute_hud_mask(prev_lvl)
                    if np.any(prev_mask):
                        return prev_mask.copy()
            return mask

        for r_idx in (0, 1, GRID_H - 2, GRID_H - 1):
            tick_steps = 0
            burst_steps = 0
            actions_seen = set()
            changed_cols = set()
            one_px_steps = 0
            for rec in active_recs:
                diff_row = rec.before[r_idx, :] != rec.after[r_idx, :]
                row_diff = int(np.count_nonzero(diff_row))
                if 1 <= row_diff <= 4:
                    tick_steps += 1
                    actions_seen.add(rec.action)
                    if row_diff == 1:
                        one_px_steps += 1
                    for c_i in np.flatnonzero(diff_row):
                        changed_cols.add(int(c_i))
                elif row_diff > 4:
                    burst_steps += 1
            tick_ratio = tick_steps / float(len(active_recs))
            if burst_steps == 0 and tick_steps >= 2 and tick_ratio >= 0.70:
                if len(actions_seen) >= 2 or tick_steps >= 3:
                    mask[r_idx, :] = True
                elif one_px_steps == tick_steps and len(changed_cols) == tick_steps:
                    cols_s = sorted(changed_cols)
                    if cols_s[-1] - cols_s[0] == len(cols_s) - 1:
                        mask[r_idx, :] = True

        for c_idx in (0, 1, GRID_W - 2, GRID_W - 1):
            tick_steps = 0
            burst_steps = 0
            actions_seen = set()
            changed_rows = set()
            one_px_steps = 0
            for rec in active_recs:
                diff_col = rec.before[:, c_idx] != rec.after[:, c_idx]
                col_diff = int(np.count_nonzero(diff_col))
                if 1 <= col_diff <= 4:
                    tick_steps += 1
                    actions_seen.add(rec.action)
                    if col_diff == 1:
                        one_px_steps += 1
                    for r_i in np.flatnonzero(diff_col):
                        changed_rows.add(int(r_i))
                elif col_diff > 4:
                    burst_steps += 1
            tick_ratio = tick_steps / float(len(active_recs))
            if burst_steps == 0 and tick_steps >= 2 and tick_ratio >= 0.70:
                if len(actions_seen) >= 2 or tick_steps >= 3:
                    mask[:, c_idx] = True
                elif one_px_steps == tick_steps and len(changed_rows) == tick_steps:
                    rows_s = sorted(changed_rows)
                    if rows_s[-1] - rows_s[0] == len(rows_s) - 1:
                        mask[:, c_idx] = True

        self._hud_mask_cache[lvl] = mask
        return mask

    def hud_report(self, level=None):
        """Human-readable HUD/budget estimate: which edge lines are masked and
        how many actions the budget bar likely has left."""
        lvl = self._current_level if level is None else int(level)
        mask = self.compute_hud_mask(lvl)
        if not np.any(mask):
            return "[hud] No ticking edge HUD detected on this level."
        rows = [r for r in (0, 1, GRID_H - 2, GRID_H - 1) if np.all(mask[r, :])]
        cols = [c for c in (0, 1, GRID_W - 2, GRID_W - 1) if np.all(mask[:, c])]
        recs = [
            r for r in self._records
            if r.level == lvl and not r.level_completed and not r.game_over
            and not r.automatic
        ]
        est = ""
        if recs and rows:
            line = rows[0]
            counts = [int(np.count_nonzero(rec.before[line, :] != 0)) for rec in recs]
            if len(counts) >= 2 and counts[0] > counts[-1] >= 0:
                per_action = (counts[0] - counts[-1]) / max(1, len(recs) - 1)
                if per_action > 0 and counts[-1] > 0:
                    est = (
                        f" Budget bar drains ~{per_action:.2f} px/action; "
                        f"~{int(counts[-1] / per_action)} actions left before it empties."
                    )
        return (
            f"[hud] Masked edge lines: rows={rows}, cols={cols}. "
            f"Excluded from physics checks and search state keys.{est}"
        )

    def summary(self, level=None):
        lvl = self._current_level if level is None else int(level)
        recs = self.transitions(level=lvl)
        by_action = {}
        for r in recs:
            tot, noops, deaths = by_action.get(r.action, (0, 0, 0))
            by_action[r.action] = (tot + 1, noops + int(r.is_noop), deaths + int(r.game_over))
        lines = [
            f"[wmlib] Level {lvl}, attempt {self._current_attempt}: "
            f"{len(recs)} transitions recorded."
        ]
        for act, (tot, noops, deaths) in sorted(by_action.items()):
            lines.append(f"  {act}: total={tot}, noop={noops}, game_over={deaths}")
        hud = self.compute_hud_mask(lvl)
        if np.any(hud):
            rows = [r for r in (0, 1, GRID_H - 2, GRID_H - 1) if np.all(hud[r, :])]
            cols = [c for c in (0, 1, GRID_W - 2, GRID_W - 1) if np.all(hud[:, c])]
            lines.append(f"  Auto-detected HUD timer mask: rows={rows}, cols={cols}")
        win_rec = self.win_transition(lvl)
        if win_rec is not None:
            lines.append(f"  Winning transition: seq_id=#{win_rec.seq_id}, action={win_rec.action}")
        return "\n".join(lines)
