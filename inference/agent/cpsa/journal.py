"""cpsa.journal — thread/process-safe on-disk transition journal with .npy cache & lazy anim frames.

Self-contained: stdlib + numpy + cpsa.wmlib/world_model/invariant_store.
"""

import os
import json
import threading
import numpy as np

from .wmlib import WorldModelLibrary
from .world_model import FalsifiedGoalLedger
from .invariant_store import GameInvariantStore

_GRID_MEM_CACHE = {}


def _grid_key(seq_id, suffix):
    return f"g{int(seq_id):07d}{suffix}"


def _atomic_write_json(path, data):
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
        fh.flush()
    os.replace(tmp, path)


def _load_grid_cached(grids_dir, key):
    ck = (grids_dir, key)
    arr = _GRID_MEM_CACHE.get(ck)
    if arr is None:
        p = os.path.join(grids_dir, key + ".npy")
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            return np.zeros((64, 64), dtype=np.int8)
        arr = np.load(p, allow_pickle=False)
        arr.flags.writeable = False
        _GRID_MEM_CACHE[ck] = arr
    return arr


class _LazyAnimFrames(list):
    def __init__(self, grids_dir, keys):
        super().__init__()
        self._grids_dir = grids_dir
        self._keys = list(keys or [])
        self._loaded = False

    def _ensure(self):
        if not self._loaded:
            super().extend(_load_grid_cached(self._grids_dir, k) for k in self._keys)
            self._loaded = True

    def __len__(self):
        return len(self._keys) if not self._loaded else super().__len__()

    def __bool__(self):
        return bool(self._keys) if not self._loaded else bool(super().__len__())

    def __getitem__(self, idx):
        self._ensure()
        return super().__getitem__(idx)

    def __iter__(self):
        self._ensure()
        return super().__iter__()


class CpsaJournalWriter:
    """Orchestrator-side owner of the journal (authoritative)."""

    def __init__(self, journal_dir):
        self.dir = str(journal_dir)
        self.grids_dir = os.path.join(self.dir, "grids")
        os.makedirs(self.grids_dir, exist_ok=True)
        self.library = WorldModelLibrary()
        self._jsonl_path = os.path.join(self.dir, "journal.jsonl")
        self._fh = None
        self._lock = threading.Lock()
        if os.path.exists(self._jsonl_path):
            self.library = load_library(self.dir)

    def _append(self, event):
        line = json.dumps(event, ensure_ascii=False) + "\n"
        with self._lock:
            if self._fh is None:
                self._fh = open(self._jsonl_path, "a", encoding="utf-8")
            self._fh.write(line)
            self._fh.flush()

    def _save_grid(self, key, grid):
        arr = np.asarray(grid, dtype=np.int8)
        final_path = os.path.join(self.grids_dir, key + ".npy")
        tmp_path = os.path.join(
            self.grids_dir, f"{key}.tmp.{os.getpid()}.{threading.get_ident()}.npy"
        )
        with open(tmp_path, "wb") as f:
            np.save(f, arr, allow_pickle=False)
            f.flush()
        os.replace(tmp_path, final_path)

    def on_level_start(self, level, start_grid):
        key = f"gstart{int(level):03d}"
        self._save_grid(key, start_grid)
        self._append({"type": "level_start", "level": int(level), "grid": key})
        self.library.on_level_start(level, start_grid)

    def on_attempt_reset(self):
        self._append({"type": "attempt_reset"})
        self.library.on_attempt_reset()

    def record_transition(self, action, before, after, anim_frames=None,
                          level_completed=False, game_over=False,
                          level_override=None, automatic=False, action_num=None):
        rec = self.library.record_transition(
            action=action, before=before, after=after, anim_frames=anim_frames,
            level_completed=level_completed, game_over=game_over,
            level_override=level_override, automatic=automatic,
        )
        seq = rec.seq_id
        self._save_grid(_grid_key(seq, "b"), rec.before)
        self._save_grid(_grid_key(seq, "a"), rec.after)
        anim_keys = []
        for i, frame in enumerate(rec.anim_frames):
            k = _grid_key(seq, f"f{i}")
            self._save_grid(k, frame)
            anim_keys.append(k)
        self._append({
            "type": "transition", "seq_id": seq, "level": rec.level,
            "attempt": rec.attempt, "step_in_attempt": rec.step_in_attempt,
            "action": rec.action, "before": _grid_key(seq, "b"),
            "after": _grid_key(seq, "a"), "anim": anim_keys,
            "num_anim_frames": rec.num_anim_frames,
            "level_completed": rec.level_completed, "game_over": rec.game_over,
            "automatic": rec.automatic, "action_num": action_num,
        })
        return rec

    def close(self):
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def load_library(journal_dir):
    """Rebuild a WorldModelLibrary from disk (sandbox side / resume)."""
    lib = WorldModelLibrary()
    grids_dir = os.path.join(journal_dir, "grids")
    path = os.path.join(journal_dir, "journal.jsonl")
    if not os.path.exists(path):
        return lib

    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except Exception:
        return lib

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
            kind = ev.get("type")
            if kind == "level_start":
                lib.on_level_start(ev["level"], _load_grid_cached(grids_dir, ev["grid"]))
            elif kind == "attempt_reset":
                lib.on_attempt_reset()
            elif kind == "transition":
                before = _load_grid_cached(grids_dir, ev["before"])
                after = _load_grid_cached(grids_dir, ev["after"])
                anim_keys = ev.get("anim") or []
                rec = lib.record_transition(
                    action=ev["action"], before=before, after=after,
                    anim_frames=None,
                    level_completed=bool(ev.get("level_completed")),
                    game_over=bool(ev.get("game_over")),
                    level_override=ev.get("level"),
                    automatic=bool(ev.get("automatic")),
                )
                rec.num_anim_frames = int(ev.get("num_anim_frames") or max(1, len(anim_keys)))
                if anim_keys:
                    rec.anim_frames = _LazyAnimFrames(grids_dir, anim_keys)
        except Exception:
            continue
    return lib


def save_ledger(journal_dir, ledger):
    _atomic_write_json(os.path.join(journal_dir, "ledger.json"), ledger.to_dict())


def load_ledger(journal_dir):
    ledger = FalsifiedGoalLedger()
    path = os.path.join(journal_dir, "ledger.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                ledger.load_dict(json.load(fh))
        except Exception:
            pass
    return ledger


def save_invariants(journal_dir, store):
    _atomic_write_json(os.path.join(journal_dir, "invariants.json"), store.to_dict())


def load_invariants(journal_dir):
    store = GameInvariantStore()
    path = os.path.join(journal_dir, "invariants.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                store.load_dict(json.load(fh))
        except Exception:
            pass
    return store
