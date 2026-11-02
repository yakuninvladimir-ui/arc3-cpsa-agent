"""cpsa.guards — orchestrator-side anti-grind guards + escalation ladder.

Runs in the ORCHESTRATOR process only.
"""

import os


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


CYCLE_REPEAT_MAX = _env_int("ARC3_CPSA_CYCLE_REPEAT_MAX", 3)
CLICK_REPEAT_MAX = _env_int("ARC3_CPSA_CLICK_REPEAT_MAX", 25)
CLICK_COARSE_MAX = _env_int("ARC3_CPSA_CLICK_COARSE_MAX", 200)
RESET_THRASH = _env_int("ARC3_CPSA_RESET_THRASH", 6)
DISTINCT_ACTION_MIN = _env_int("ARC3_CPSA_DISTINCT_ACTION_MIN", 40)
ESCALATE_T1 = _env_int("ARC3_CPSA_ESCALATE_T1", 120)
ESCALATE_T2 = _env_int("ARC3_CPSA_ESCALATE_T2", 250)
ESCALATE_FLOOR = _env_int("ARC3_CPSA_ESCALATE_FLOOR", 400)

_CYCLE_DETAIL = (
    "CPSA CYCLE GUARD: the pair (this exact board state, {sig}) has already been "
    "executed {count} times on this level. The environment is deterministic: "
    "repeating it cannot produce new information. "
    "The action was NOT executed and no actions were spent. Ways out: (1) update "
    "simulate()/is_goal (or project(frame)) and run check_sim — the mismatch you "
    "keep hitting IS the counterexample; (2) search(current_frame, simulate, is_goal, "
    "actions=...) for a different route; (3) try a DIFFERENT action from this state; "
    "(4) inspect wmlib.transitions(level=...) and the animation frames of the "
    "transitions your model does not reproduce. If you have a specific reason to "
    "believe hidden state changed, issue the exact same action again immediately — "
    "that overrides the block and spends one action."
)

_CLICK_DETAIL = (
    "CPSA CLICK GUARD: MOUSE(row={r}, col={c}) has been clicked {count} times on "
    "this level without completing it. Not executed; no actions spent. Re-clicking "
    "the same pixel is not an experiment. Ways out: model what this cell is "
    "(finders over colors/segmentation), encode its rule in simulate(), verify "
    "with check_sim, then search() with actions=candidate_clicks(current_frame). "
    "Issuing the exact same click again immediately overrides the block."
)

_SWEEP_DETAIL = (
    "CPSA SWEEP GUARD: {count} clicks inside the 4x4 block starting at "
    "row={br}, col={bc} on this level without completion — this looks like a "
    "coordinate sweep, which RHAE punishes quadratically. Not executed; no "
    "actions spent. Stop sweeping: parse the objects (colors/cells/bbox/"
    "segmentation), promote rules into simulate(), pass check_sim, and let "
    "search() choose clicks. Issuing the same click again immediately overrides "
    "the block, but each insistence costs one scored action."
)

ESCALATION_T1 = """=== CPSA ESCALATION T1 (level {level}: {spent} actions without completion) ===
This level is not yielding to live play. Binding directive until it completes:
1. Inventory BOTH what wmlib leaves unexplained AND the reachable places/states you have never visited (save_invariants(hazard_or_falsified='unexplored: ...')). Unexplored territory outranks new-mechanic guesses.
2. If pixel-level simulation is complex, define a state projector `def project(frame):` that extracts only the essential object positions/states (objects_of(frame) / find_color), write `simulate(state, action)` and `is_goal(state)`, and verify with `check_sim(simulate, is_goal, project=project)`.
3. plan = search(current_frame, simulate, is_goal, actions=..., project=project) with bounded max_states; then act_checked(plan, simulate, is_goal, project=project).
4. Spend single actions ONLY as discriminating probes between surviving hypotheses.
"""

ESCALATION_T2 = """=== CPSA ESCALATION T2 (level {level}: {spent} actions, {resets} self-resets) ===
Still stuck after simulating. Assume one of your rules is WRONG or a region is unvisited:
enumerate the reachable-state frontier under your model, prefer plans that reach board configurations never observed in wmlib, re-derive any rule that claims the goal is unreachable, and inspect the animation frames (last_animation_frames / wmlib transitions with num_anim_frames>1) of every transition your model does not reproduce.
"""

SCORE_FLOOR = """=== CPSA SCORE FLOOR (level {level}: {spent} actions) ===
This level has absorbed {spent} actions; with the quadratic RHAE penalty its points are largely gone (~4% of par at 5x baseline). Completion still matters (an incomplete level caps the whole game): finish it with the CHEAPEST verified plan you can search, stop probing here, and invest the remaining budget into the world model that makes the LATER, higher-weighted levels cheap.
"""


class CpsaGuards:
    def __init__(self, interior_hash_fn):
        self._hash = interior_hash_fn
        self._cycle_counts = {}
        self._click_counts = {}
        self._coarse_counts = {}
        self._armed = None
        self._level = None
        self._level_actions = 0
        self._level_self_resets = 0
        self._level_distinct_sigs = set()

    def note_level(self, level):
        if level != self._level:
            self._level = level
            self._cycle_counts.clear()
            self._click_counts.clear()
            self._coarse_counts.clear()
            self._armed = None
            self._level_actions = 0
            self._level_self_resets = 0
            self._level_distinct_sigs = set()

    def note_executed_action(self, level, model_action_name, action_data=None):
        """Called by the orchestrator AFTER an action actually executed."""
        self.note_level(level)
        self._level_actions += 1
        sig = self._sig(model_action_name, action_data)
        self._level_distinct_sigs.add(sig)
        if str(model_action_name).upper() == "RESET":
            self._level_self_resets += 1

    @staticmethod
    def _sig(model_action_name, action_data):
        name = str(model_action_name).upper()
        if name == "MOUSE" and action_data:
            return f"MOUSE({int(action_data.get('row', -1))},{int(action_data.get('col', -1))})"
        return name

    def check(self, grid_rows, model_action_name, action_data, level):
        """Return (stop_reason, detail) to refuse BEFORE the engine step
        (0 actions spent), or None to allow."""
        self.note_level(level)
        name = str(model_action_name).upper()
        sig = self._sig(name, action_data)
        try:
            state_hash = self._hash([list(row) for row in grid_rows]) if grid_rows else None
        except Exception:
            state_hash = None

        if self._armed is not None:
            kind, key = self._armed
            self._armed = None
            matches = (
                (kind == "cycle" and key == (level, state_hash, sig))
                or (kind == "click" and key == self._click_key(name, action_data, level))
                or (kind == "sweep" and key == self._coarse_key(name, action_data, level))
            )
            if matches:
                if kind == "cycle":
                    self._cycle_counts[key] = 1
                return None

        if state_hash is not None:
            key = (level, state_hash, sig)
            count = self._cycle_counts.get(key, 0)
            if count >= CYCLE_REPEAT_MAX:
                self._armed = ("cycle", key)
                return ("cpsa_cycle_repeat", _CYCLE_DETAIL.format(sig=sig, count=count))
            self._cycle_counts[key] = count + 1

        if name == "MOUSE" and action_data:
            ck = self._click_key(name, action_data, level)
            if ck is not None:
                count = self._click_counts.get(ck, 0)
                if count >= CLICK_REPEAT_MAX:
                    self._armed = ("click", ck)
                    return ("cpsa_click_repeat", _CLICK_DETAIL.format(r=ck[1], c=ck[2], count=count))
                self._click_counts[ck] = count + 1
            gk = self._coarse_key(name, action_data, level)
            if gk is not None:
                count = self._coarse_counts.get(gk, 0)
                if count >= CLICK_COARSE_MAX:
                    self._armed = ("sweep", gk)
                    return ("cpsa_click_sweep", _SWEEP_DETAIL.format(count=count, br=gk[1] * 4, bc=gk[2] * 4))
                self._coarse_counts[gk] = count + 1
        return None

    @staticmethod
    def _click_key(name, action_data, level):
        if name != "MOUSE" or not action_data:
            return None
        try:
            return (level, int(action_data.get("row")), int(action_data.get("col")))
        except Exception:
            return None

    @staticmethod
    def _coarse_key(name, action_data, level):
        if name != "MOUSE" or not action_data:
            return None
        try:
            r = int(action_data.get("row")) // 4
            c = int(action_data.get("col")) // 4
            return (level, r, c)
        except Exception:
            return None

    def reset_thrash_active(self):
        return (
            self._level_self_resets >= RESET_THRASH
            and len(self._level_distinct_sigs) <= DISTINCT_ACTION_MIN
        )

    def directive(self, level):
        self.note_level(level)
        spent = self._level_actions
        resets = self._level_self_resets
        parts = []
        if spent >= ESCALATE_FLOOR:
            parts.append(SCORE_FLOOR.format(level=level, spent=spent))
        if spent >= ESCALATE_T2 or self.reset_thrash_active():
            parts.append(ESCALATION_T2.format(level=level, spent=spent, resets=resets))
        elif spent >= ESCALATE_T1:
            parts.append(ESCALATION_T1.format(level=level, spent=spent))
        return "\n".join(parts)

    @property
    def level_actions(self):
        return self._level_actions

    @property
    def level_self_resets(self):
        return self._level_self_resets
