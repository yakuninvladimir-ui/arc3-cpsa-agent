"""cpsa.invariant_store — persistent cross-level knowledge + reasoning lifecycle.

Self-contained: stdlib only. No __future__ imports.
"""

import time

VALID_TAGS = ("checked", "assumed")
_REASONING_KEYS = ("reasoning_content", "reasoning")


def _split_tag(text):
    """Normalize a mechanics/invariant string to '[tag] text'."""
    s = str(text).strip()
    for tag in VALID_TAGS:
        if s.lower().startswith(f"[{tag}]"):
            return tag, s[len(tag) + 2:].strip()
    return "assumed", s


class GameInvariantStore:
    """Proven rules, win conditions, falsified hypotheses. Survives deaths and
    level changes for the whole game."""

    def __init__(self):
        self.confirmed_mechanics = {}
        self.win_condition = ""
        self.falsified_hypotheses_and_hazards = []
        self.level_deltas = {}

    def update_invariants(self, mechanics=None, win_condition=None,
                          hazard_or_falsified=None, level=None, level_delta=None,
                          allow_checked_mechanics=True, allow_checked_win=True):
        """Update persistent store. Enforces MBEL §2.1 Non-Promotion & LAF §12
        write-back guard: '[checked]' tags are downgraded to '[assumed]' when
        the caller has not certified the claim via check_sim / observed FOLLOW."""
        if mechanics:
            for k, v in mechanics.items():
                tag, clean = _split_tag(v)
                if tag == "checked" and not allow_checked_mechanics:
                    tag = "assumed"
                self.confirmed_mechanics[str(k).strip()] = f"[{tag}] {clean}"
        if win_condition is not None and str(win_condition).strip():
            tag, clean = _split_tag(win_condition)
            if tag == "checked" and not allow_checked_win:
                tag = "assumed"
            self.win_condition = f"[{tag}] {clean}"
        if hazard_or_falsified is not None and str(hazard_or_falsified).strip():
            entry = str(hazard_or_falsified).strip()
            if entry not in self.falsified_hypotheses_and_hazards:
                self.falsified_hypotheses_and_hazards.append(entry)
                if len(self.falsified_hypotheses_and_hazards) > 20:
                    self.falsified_hypotheses_and_hazards = self.falsified_hypotheses_and_hazards[-20:]
        if level is not None and level_delta is not None and str(level_delta).strip():
            self.level_deltas[int(level)] = str(level_delta).strip()
        return self.render_for_prompt(current_level=level)

    def record_auto_falsification(self, reason):
        clean = str(reason).strip().replace("\n", " | ")
        if clean and clean not in self.falsified_hypotheses_and_hazards:
            self.falsified_hypotheses_and_hazards.append(clean[:320])
            if len(self.falsified_hypotheses_and_hazards) > 20:
                self.falsified_hypotheses_and_hazards = self.falsified_hypotheses_and_hazards[-20:]

    def to_dict(self):
        return {
            "confirmed_mechanics": dict(self.confirmed_mechanics),
            "win_condition": self.win_condition,
            "falsified_hypotheses_and_hazards": list(self.falsified_hypotheses_and_hazards),
            "level_deltas": {str(k): v for k, v in self.level_deltas.items()},
        }

    def load_dict(self, data):
        if not data:
            return
        self.confirmed_mechanics = {
            str(k): str(v) for k, v in data.get("confirmed_mechanics", {}).items()
        }
        self.win_condition = str(data.get("win_condition", ""))
        self.falsified_hypotheses_and_hazards = [
            str(x) for x in data.get("falsified_hypotheses_and_hazards", [])
        ]
        self.level_deltas = {
            int(k): str(v) for k, v in data.get("level_deltas", {}).items()
        }

    def merge_dict(self, data):
        """Merge sandbox-returned state into the orchestrator copy."""
        if not data:
            return
        incoming_falsified = [str(x) for x in data.get("falsified_hypotheses_and_hazards", [])]
        merged = list(self.falsified_hypotheses_and_hazards)
        for item in incoming_falsified:
            if item not in merged:
                merged.append(item)
        self.load_dict(data)
        self.falsified_hypotheses_and_hazards = merged[-20:]

    def render_for_prompt(self, current_level=None):
        lines = ["=== PERSISTENT GAME INVARIANT STORE (SURVIVES DEATHS & LEVELS) ==="]
        lines.append(
            "[1. CONFIRMED MECHANICS]  ([checked]=verified by check_sim/observations, "
            "[assumed]=hypothesis — never build multi-step plans on assumed)"
        )
        if self.confirmed_mechanics:
            for k, v in self.confirmed_mechanics.items():
                lines.append(f"  - {k}: {v}")
        else:
            lines.append("  (None recorded yet — call save_invariants(mechanics={...}) once verified)")
        lines.append("[2. CONFIRMED WIN CONDITION]")
        lines.append(
            f"  {self.win_condition or '(Not yet confirmed — derive it from what changes at frames where levels END: wmlib.win_transition())'}"
        )
        lines.append("[3. FALSIFIED HYPOTHESES & KNOWN HAZARDS (DO NOT REPEAT)]")
        if self.falsified_hypotheses_and_hazards:
            for h in self.falsified_hypotheses_and_hazards[-10:]:
                lines.append(f"  ! {h}")
        else:
            lines.append("  (None)")
        if self.level_deltas:
            lines.append("[4. LEVEL PROGRESSION DELTAS]")
            for lvl, delta in sorted(self.level_deltas.items()):
                marker = " <-- CURRENT" if current_level == lvl else ""
                lines.append(f"  - Level {lvl}{marker}: {delta}")
        lines.append("==================================================================")
        return "\n".join(lines)


class ReasoningLifecycleManager:
    """Retention + falsification-triggered amputation of reasoning traces
    (supports both `reasoning` and `reasoning_content` message keys)."""

    FALSIFICATION_CLASSES = (
        "EXPECTATION_MISMATCH", "HIDDEN_ANIMATION", "SIM_DIVERGENCE",
        "GOAL_FALSIFIED", "GAME_OVER", "SIM_ERROR",
    )

    def __init__(self, keep_last_k=6, min_interval_s=0.0):
        self.keep_last_k = max(1, int(keep_last_k))
        self.min_interval_s = float(min_interval_s)
        self._last_amputation_ts = 0.0
        self._last_amputation_class = None

    @staticmethod
    def classify(reason):
        text = str(reason or "")
        for cls in ReasoningLifecycleManager.FALSIFICATION_CLASSES:
            if cls in text:
                return cls
        return "OTHER"

    def prune_stale_reasoning(self, history_messages):
        """Keep reasoning/reasoning_content only on the last keep_last_k assistant msgs."""
        assistant_indices = [
            i for i, m in enumerate(history_messages)
            if isinstance(m, dict) and m.get("role") == "assistant"
        ]
        if len(assistant_indices) <= self.keep_last_k:
            return 0
        stripped = 0
        for idx in assistant_indices[:-self.keep_last_k]:
            msg = history_messages[idx]
            had = False
            for k in _REASONING_KEYS:
                if k in msg:
                    msg.pop(k, None)
                    had = True
            if had:
                stripped += 1
        return stripped

    def should_amputate(self, reason):
        if self.min_interval_s <= 0:
            return True
        return (time.monotonic() - self._last_amputation_ts) >= self.min_interval_s

    def amputate_all_reasoning_on_falsification(self, history_messages, reason=""):
        """Remove reasoning/reasoning_content from ALL prior assistant messages.
        Returns (amputated_count, was_rate_limited)."""
        if not self.should_amputate(reason):
            return 0, True
        amputated = 0
        for m in history_messages:
            if isinstance(m, dict) and m.get("role") == "assistant":
                had = False
                for k in _REASONING_KEYS:
                    if k in m:
                        m.pop(k, None)
                        had = True
                if had:
                    amputated += 1
        self._last_amputation_ts = time.monotonic()
        self._last_amputation_class = self.classify(reason)
        return amputated, False
