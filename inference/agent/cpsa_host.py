"""CpsaHost — orchestrator-side CPSA 2.0 coordinator (lives in ToolAgent).

Location: ARC3-Inference/inference/agent/cpsa_host.py
"""

import os
from pathlib import Path

from inference.agent.cpsa import journal as cpsa_journal
from inference.agent.cpsa.guards import CpsaGuards
from inference.agent.cpsa.invariant_store import (
    GameInvariantStore, ReasoningLifecycleManager,
)
from inference.agent.cpsa.world_model import FalsifiedGoalLedger
from inference.agent.cpsa.consolidation import check_level_consolidated
from inference.agent.cpsa.prompts import (
    CONSOLIDATION_PROMPT_TEMPLATE, FALSIFICATION_INTERRUPT_TEMPLATE,
)


def cpsa_enabled():
    return os.environ.get("ARC3_CPSA", "0").strip().lower() in {"1", "on", "true", "yes"}


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _env_bool(name, default=False):
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "on", "true", "yes"}


class CpsaHost:
    def __init__(self, interior_hash_fn):
        self.journal = None
        self.journal_dir = None
        self._raw_hash_fn = interior_hash_fn
        self.guards = CpsaGuards(self._hud_aware_hash)
        self.store = GameInvariantStore()
        self.ledger = FalsifiedGoalLedger()
        self.reasoning = ReasoningLifecycleManager(
            keep_last_k=_env_int("ARC3_REASONING_KEEP_LAST_K", 6),
            min_interval_s=float(os.environ.get("ARC3_CPSA_AMPUTATE_MIN_INTERVAL_S", "60")),
        )
        self.last_completed_level = None
        self.consolidation = None  # {"level": int, "turns_used": int}
        self._started_levels = set()
        self._last_prompt_level = None

    # ------------------------------------------------------------- lifecycle
    def attach(self, state_path):
        sp = Path(state_path) if state_path else Path("/tmp")
        stem = sp.stem if sp.stem and sp.stem != "runtime_state" else f"{sp.parent.name}_{id(self):x}"
        self.journal_dir = str(sp.parent / f"cpsa_{stem}")
        self.journal = cpsa_journal.CpsaJournalWriter(self.journal_dir)
        self.store = cpsa_journal.load_invariants(self.journal_dir)
        self.ledger = cpsa_journal.load_ledger(self.journal_dir)

    def close(self):
        if self.journal is not None:
            self.journal.close()

    def pkg_parent(self):
        """Directory that must go on the sandbox sys.path so `import cpsa.*`
        works there: the parent of the cpsa package (= inference/agent)."""
        import inference.agent.cpsa as pkg
        return str(Path(pkg.__file__).resolve().parent.parent)

    def sandbox_config(self):
        return {
            "enabled": True,
            "journal_dir": self.journal_dir,
            "consolidation_readonly": bool(self.consolidation),
        }

    # ------------------------------------------------------- journal (solver)
    def ensure_level_started(self, level, grid):
        if level is None or grid is None:
            return
        if level not in self._started_levels:
            self._started_levels.add(level)
            if self.journal.library.current_level != level or not self.journal.library._records:
                self.journal.on_level_start(level, grid)

    def _hud_aware_hash(self, grid):
        """Mask out any detected HUD timer cells before computing state hash for
        CpsaGuards so ticking HUD bars do not blind cycle/noop detection."""
        if self.journal is not None and grid:
            try:
                import numpy as np
                hud = self.journal.library.compute_hud_mask()
                if np.any(hud):
                    arr = np.asarray(grid, dtype=np.int16).copy()
                    if arr.shape == hud.shape:
                        arr[hud] = 0
                        return self._raw_hash_fn(arr.tolist())
            except Exception:
                pass
        return self._raw_hash_fn(grid)

    def on_transition(self, ev):
        """Solver hook: one executed engine action (including automatic resets)."""
        lvl_before = ev.get("level_before")
        lvl_after = ev.get("level_after")
        display = str(ev.get("action_display") or "")
        is_reset = display.upper().startswith("RESET") or bool(ev.get("automatic"))
        self.ensure_level_started(lvl_before, ev.get("before"))
        self.journal.record_transition(
            action=display,
            before=ev.get("before"),
            after=ev.get("after"),
            anim_frames=ev.get("anim_chain") or [],
            level_completed=bool(ev.get("level_completed")),
            game_over=bool(ev.get("game_over")),
            level_override=lvl_before,
            automatic=is_reset,
            action_num=ev.get("action_num"),
        )
        if ev.get("level_completed") and not ev.get("run_complete"):
            self.last_completed_level = lvl_before
            if lvl_after is not None:
                self.ensure_level_started(lvl_after, ev.get("after"))
        if bool(ev.get("game_over")) or is_reset:
            self.journal.on_attempt_reset()
            if display.upper().startswith("RESET") and not ev.get("automatic"):
                self.guards.note_executed_action(lvl_before, "RESET")

    # ---------------------------------------------------------------- guards
    def guard_check(self, grid, engine_action_name, action_data, level):
        """Per-position hook (same shape as solver.action_guard_hook)."""
        from inference.agent.action_names import to_model_action
        model_name = to_model_action(engine_action_name)
        data = dict(action_data or {})
        if model_name == "MOUSE" and "row" not in data:
            if "y" in data:
                data["row"] = data.get("y")
            if "x" in data:
                data["col"] = data.get("x")
        rows = [list(r) for r in grid] if grid else []
        return self.guards.check(rows, model_name, data, level)

    def note_action_call(self, compact_payload, requested_actions):
        """Escalation bookkeeping from _handle_action results."""
        if not compact_payload or not compact_payload.get("executed"):
            return
        level = compact_payload.get("level")
        try:
            count = max(1, int(compact_payload.get("executed_count") or 1))
        except Exception:
            count = 1
        names = list(requested_actions or [])
        from inference.agent.action_names import to_model_action
        for i in range(count):
            item = names[i] if i < len(names) else (names[-1] if names else {})
            name = item.get("action") if isinstance(item, dict) else str(item)
            data = item if isinstance(item, dict) else None
            model_name = to_model_action(str(name or "")) or str(name or "")
            if str(model_name).upper() != "RESET":
                self.guards.note_executed_action(level, model_name, data)

    # --------------------------------------------------------- sandbox sync
    def sync_from_sandbox(self, sandbox_result):
        """Merge sandbox-returned CPSA fields into orchestrator-owned state."""
        if not isinstance(sandbox_result, dict):
            return
        store_state = sandbox_result.get("invariant_store_state")
        if store_state:
            self.store.merge_dict(store_state)
            cpsa_journal.save_invariants(self.journal_dir, self.store)
        ledger_state = sandbox_result.get("falsified_ledger_state")
        if ledger_state:
            self.ledger.load_dict(ledger_state)
            cpsa_journal.save_ledger(self.journal_dir, self.ledger)
        reason = str(sandbox_result.get("falsification_reason") or "")
        if sandbox_result.get("hypothesis_falsified") and reason:
            self.store.record_auto_falsification(reason)
            cpsa_journal.save_invariants(self.journal_dir, self.store)

    def falsification_block(self, sandbox_result):
        """Formatted interrupt text to append to the tool output (or '')."""
        if not isinstance(sandbox_result, dict):
            return ""
        if not sandbox_result.get("hypothesis_falsified"):
            return ""
        return "\n" + FALSIFICATION_INTERRUPT_TEMPLATE.format(
            falsification_reason=str(sandbox_result.get("falsification_reason") or "")[:800]
        )

    # -------------------------------------------------------- consolidation
    def begin_consolidation(self, kept_functions=None):
        """Start a zero-action consolidation window when a level completes and
        the retained model does not yet pass the Judge on that level."""
        if os.environ.get("ARC3_CONSOLIDATION_TURN", "1") != "1":
            return ""
        if self.consolidation is not None or self.last_completed_level is None:
            return ""
        level = self.last_completed_level
        status = self.consolidation_status(kept_functions, level)
        if status.get("consolidated"):
            return ""
        self.consolidation = {"level": level, "turns_used": 0}
        next_level = self.journal.library.current_level if self.journal else level + 1
        return "\n" + CONSOLIDATION_PROMPT_TEMPLATE.format(
            level_num=level,
            next_level_num=next_level,
            win_seq_id=status.get("win_seq_id", -1),
            win_action=status.get("win_action", "?"),
        )

    def consolidation_status(self, kept_functions, level):
        return check_level_consolidated(
            self.journal.library, self.ledger, kept_functions or {}, level
        )

    def consolidation_tick(self, kept_functions):
        """Call after each readonly sandbox snippet while consolidation is active.
        Returns True while the readonly consolidation window is still running."""
        if self.consolidation is None:
            return False
        self.consolidation["turns_used"] += 1
        max_turns = _env_int("ARC3_CPSA_CONSOLIDATION_MAX_TURNS", 2)
        status = self.consolidation_status(kept_functions, self.consolidation["level"])
        if status.get("consolidated") or self.consolidation["turns_used"] >= max_turns:
            self.consolidation = None
            return False
        return True

    # ------------------------------------------------------------- prompts
    def on_prompt(self, level, grid):
        """Called from _build_user_prompt. Returns (prefix_text, clear_history).

        Injects the persistent GameInvariantStore + previous level summary +
        HUD report at level boundaries, and escalation directives when stuck.
        Preserves KV prefix cache by default (clear_history=True only when
        ARC3_CPSA_WIPE_HISTORY_ON_LEVEL=1)."""
        prefix_parts = []
        clear_history = False
        if level is not None:
            self.ensure_level_started(level, grid)
            if self._last_prompt_level is not None and level != self._last_prompt_level:
                clear_history = _env_bool("ARC3_CPSA_WIPE_HISTORY_ON_LEVEL", False)
                prev = self._last_prompt_level
                prefix_parts.append(
                    f"=== LEVEL {prev} -> {level} TRANSITION (persistent world-model memory) ===\n"
                    + self.store.render_for_prompt(current_level=level)
                    + "\n" + self.journal.library.summary(level=prev)
                    + "\n" + self.journal.library.hud_report(level)
                )
            else:
                if (
                    self.store.confirmed_mechanics
                    or self.store.win_condition
                    or self.store.falsified_hypotheses_and_hazards
                    or self.store.level_deltas
                ):
                    prefix_parts.append(self.store.render_for_prompt(current_level=level))
            self._last_prompt_level = level
            directive = self.guards.directive(level)
            if directive:
                prefix_parts.append(directive)
        return ("\n\n".join(prefix_parts) + "\n\n" if prefix_parts else ""), clear_history
