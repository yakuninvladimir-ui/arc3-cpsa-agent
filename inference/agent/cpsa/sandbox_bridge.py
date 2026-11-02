"""cpsa.sandbox_bridge — glue executed INSIDE the sandbox subprocess.

env contract (all provided by the bootstrap in python_tool_sandbox.py):
  env["enabled"]                bool
  env["journal_dir"]            str  (absolute; orchestrator-owned journal)
  env["consolidation_readonly"] bool
  env["step"](payload) -> dict  (bootstrap action())
  env["grid"]() -> grid-like    (current_frame._grid or current_frame)
  env["level"]() -> int         (current_frame.level)
  env["anim_frames"]() -> list  (FrameView list; may be [] or raise -> ignored)
  env["clear_stale"]() -> None  (optional; clears _stale_state before guarded step)

build() returns (exports: dict, get_final_fields: callable -> dict).
"""

import os
import numpy as np

from .wmlib import WorldModelLibrary
from .world_model import (
    FalsifiedGoalLedger, _extract_grid, bind_world_model_api, parse_mouse,
)
from .invariant_store import GameInvariantStore
from . import journal as journal_io
from . import helpers as helpers_mod


class _FakeRecord:
    """TransitionRecord stand-in for skipped/redundant steps (no env action)."""

    def __init__(self, grid, level, seq_id=-1):
        g = np.asarray(_extract_grid(grid), dtype=np.int8)
        self.seq_id = seq_id
        self.level = level
        self.attempt = 0
        self.step_in_attempt = -1
        self.action = "SKIPPED"
        self.before = g
        self.after = g
        self.changed_coords = []
        self.num_anim_frames = 1
        self.anim_frames = []
        self.level_completed = False
        self.game_over = False
        self.automatic = True

    @property
    def is_noop(self):
        return True

    @property
    def has_hidden_animation(self):
        return False

    @property
    def pre_level_swap_grid(self):
        return None


def _to_payload(act):
    """Canonical CPSA action string -> bootstrap action() payload.
    'MOUSE(row=15, col=20)' -> {"action": "MOUSE", "row": 15, "col": 20}."""
    if isinstance(act, dict):
        return act
    s = str(act).strip()
    rc = parse_mouse(s)
    if rc is not None:
        return {"action": "MOUSE", "row": rc[0], "col": rc[1]}
    return s


def build(env):
    enabled = bool(env.get("enabled"))
    if not enabled:
        return {}, lambda: {}

    journal_dir = str(env.get("journal_dir") or "")
    readonly = bool(env.get("consolidation_readonly"))
    step_fn = env["step"]
    grid_fn = env["grid"]
    level_fn = env["level"]
    anim_fn = env.get("anim_frames")
    clear_stale_fn = env.get("clear_stale")

    wmlib = WorldModelLibrary()
    if journal_dir and os.path.isdir(journal_dir):
        try:
            wmlib = journal_io.load_library(journal_dir)
        except Exception as e:
            print(
                f"[cpsa] WARNING: journal reload failed ({type(e).__name__}: {e}); "
                f"starting with an empty in-snippet journal."
            )
        if readonly and getattr(wmlib, "_level_win_records", None):
            wmlib._current_level = max(wmlib._level_win_records.keys())
            level_fn = lambda: wmlib.current_level
    ledger = journal_io.load_ledger(journal_dir) if journal_dir else FalsifiedGoalLedger()
    store = journal_io.load_invariants(journal_dir) if journal_dir else GameInvariantStore()

    falsified_flags = {"hit": False, "reasons": []}
    last_step_result = {"value": None}

    def _notify(reason):
        falsified_flags["hit"] = True
        falsified_flags["reasons"].append(str(reason))
        try:
            store.record_auto_falsification(str(reason))
        except Exception:
            pass

    def _current_grid_np():
        return _extract_grid(grid_fn()).astype(np.int16)

    def _step_and_mirror(act_name, clear_stale=True):
        """One environment step + local journal mirror."""
        if readonly:
            raise RuntimeError(
                "CONSOLIDATION_READONLY: the level is already completed. 0 environment "
                "actions are allowed on this turn. Only define/update simulate, project, "
                "and is_goal, run check_sim(), and call save_invariants()."
            )
        payload_act = _to_payload(act_name)
        canonical = act_name if isinstance(act_name, str) else str(act_name)
        rc = parse_mouse(canonical)
        if rc is not None:
            canonical = f"MOUSE(row={rc[0]}, col={rc[1]})"

        if canonical.upper() == "RESET" and wmlib.step_in_attempt == 0:
            print(
                "[GUARD] Board is already at the initial state of the attempt "
                "(step 0); redundant RESET skipped — 0 actions spent."
            )
            return _FakeRecord(_current_grid_np(), level_fn())

        # Clear stale state only for self-verifying calls (act_expect, act_checked).
        # Raw action() calls preserve _stale_state to prevent infinite no-op loops.
        if clear_stale and callable(clear_stale_fn):
            try:
                clear_stale_fn()
            except Exception:
                pass

        before = _current_grid_np()
        lvl_before = level_fn()
        result = step_fn([payload_act] if isinstance(payload_act, dict) else payload_act)
        last_step_result["value"] = result
        if not isinstance(result, dict):
            raise RuntimeError("CPSA: unexpected action() result type")
        if result.get("executed") is False and result.get("stop_reason"):
            raise RuntimeError(f"{result.get('stop_reason')}: {result.get('stop_detail') or ''}")
        after = _current_grid_np()
        lvl_done = bool(result.get("level_completed"))
        is_dead = bool(result.get("game_over"))
        run_done = bool(result.get("run_complete") or result.get("done"))

        anim = []
        try:
            if anim_fn is not None and (result.get("animation") or lvl_done):
                raw = anim_fn() or []
                anim = [np.asarray(_extract_grid(f), dtype=np.int8) for f in raw]
        except Exception:
            anim = []

        rec = wmlib.record_transition(
            action=canonical, before=before, after=after, anim_frames=anim,
            level_completed=lvl_done or run_done, game_over=is_dead,
            level_override=lvl_before,
        )
        if is_dead:
            _notify(f"GAME_OVER on action {canonical}")
            wmlib.on_attempt_reset()
        elif canonical.upper() == "RESET":
            wmlib.on_attempt_reset()
        if lvl_done and not run_done:
            try:
                wmlib.on_level_start(level_fn(), after)
            except Exception:
                pass
        return rec

    api = bind_world_model_api(
        wmlib=wmlib,
        falsified_ledger=ledger,
        step_single_action_fn=_step_and_mirror,
        get_current_grid_fn=_current_grid_np,
        get_current_level_fn=level_fn,
        notify_hypothesis_falsified_fn=_notify,
    )

    # Track authoritative verification state inside this snippet (MBEL §2.1 / LAF §12 C2 guard)
    cert_state = {
        "check_sim_called": False,
        "check_sim_ok": False,
        "goal_confirmed": False,
        "act_expect_follow": False,
    }

    _orig_check_sim = api["check_sim"]
    def _tracked_check_sim(*args, **kwargs):
        res = _orig_check_sim(*args, **kwargs)
        cert_state["check_sim_called"] = True
        cert_state["check_sim_ok"] = bool(res.get("ok") or res.get("advisory_ok"))
        cert_state["goal_confirmed"] = (res.get("goal_status") == "CONFIRMED")
        return res
    api["check_sim"] = _tracked_check_sim

    _orig_act_expect = api["act_expect"]
    def _tracked_act_expect(*args, **kwargs):
        res = _orig_act_expect(*args, **kwargs)
        if isinstance(res, dict) and res.get("verdict_v4") == "FOLLOW":
            cert_state["act_expect_follow"] = True
        return res
    api["act_expect"] = _tracked_act_expect

    def save_invariants(mechanics=None, win_condition=None, hazard_or_falsified=None,
                        level=None, level_delta=None):
        lvl = wmlib.current_level if level is None else int(level)
        has_any_win = any(
            wmlib.win_transition(l) is not None for l in range(1, max(1, lvl) + 1)
        )
        has_active_obs = any(
            not r.is_noop for r in wmlib.transitions(level=lvl, include_level_completed=True)
        ) or has_any_win
        # Mechanics may be tagged [checked] only if check_sim passed (or was not failed and
        # an observed FOLLOW transition / act_expect predicate succeeded).
        if cert_state["check_sim_called"]:
            allow_mech = cert_state["check_sim_ok"]
        else:
            allow_mech = bool(cert_state["act_expect_follow"] or has_active_obs)
        # Win condition may be tagged [checked] only if a level win was actually observed
        # and check_sim did not contradict the goal.
        allow_win = bool(
            has_any_win and (not cert_state["check_sim_called"] or cert_state["check_sim_ok"])
        )
        rendered = store.update_invariants(
            mechanics=mechanics, win_condition=win_condition,
            hazard_or_falsified=hazard_or_falsified, level=lvl,
            level_delta=level_delta,
            allow_checked_mechanics=allow_mech,
            allow_checked_win=allow_win,
        )
        print(rendered)
        return rendered

    def _guarded_action(actions, reasoning=None):
        """Wrap raw action() so direct calls respect CONSOLIDATION_READONLY and
        mirror every executed step into the in-snippet wmlib journal."""
        if readonly:
            raise RuntimeError(
                "CONSOLIDATION_READONLY: the level is already completed. 0 environment "
                "actions are allowed on this turn. Only define/update simulate, project, "
                "and is_goal, run check_sim(), and call save_invariants()."
            )
        if isinstance(actions, (str, dict)):
            seq = [actions]
        else:
            seq = list(actions)
        last_res = None
        for act_item in seq:
            _step_and_mirror(act_item, clear_stale=False)
            last_res = last_step_result["value"]
        return last_res

    exports = dict(api)
    exports.update(helpers_mod.HELPER_EXPORTS)
    exports["wmlib"] = wmlib
    exports["save_invariants"] = save_invariants
    exports["FalsifiedGoalLedger"] = FalsifiedGoalLedger
    exports["GameInvariantStore"] = GameInvariantStore
    exports["action"] = _guarded_action
    exports["np"] = np

    def get_final_fields():
        summary = ""
        try:
            summary = wmlib.summary()
        except Exception:
            pass
        return {
            "hypothesis_falsified": bool(falsified_flags["hit"]),
            "falsification_reason": "\n".join(falsified_flags["reasons"])[:2000],
            "invariant_store_state": store.to_dict(),
            "falsified_ledger_state": ledger.to_dict(),
            "wmlib_summary": summary[:2000],
            "cpsa_consolidation_readonly": readonly,
        }

    return exports, get_final_fields
