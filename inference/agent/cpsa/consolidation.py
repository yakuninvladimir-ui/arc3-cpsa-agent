"""cpsa.consolidation — orchestrator-side, in-process level consolidation check.

Runs the SAME V4 Judge (cpsa.world_model.evaluate_model) inside the harness
process — no sandbox subprocess, no environment actions. Supports both Grid
mode (simulate + is_goal) and Boundary Projection mode (project + simulate + is_goal).
"""

import numpy as np

from .world_model import SimFrame, FalsifiedGoalLedger, evaluate_model


def build_model_namespace(simulate_source, is_goal_source, project_source=None, extra_sources=(), library=None):
    """Exec retained model sources in an isolated namespace with the CPSA API
    available. Returns (namespace, error_string|None)."""
    from . import wmlib as wmlib_mod
    from . import world_model as wm_mod
    from . import helpers as helpers_mod

    ns = {"np": np, "numpy": np, "__builtins__": __builtins__}
    if library is not None:
        ns["wmlib"] = library
    ns.update({k: getattr(wm_mod, k) for k in (
        "SimFrame", "UNKNOWN_CELL", "copy_state", "diff_text", "cells", "bbox",
        "colors", "parse_mouse", "candidate_clicks", "ARC_COLOR_CHARS",
        "resolve_color", "color_char",
    )})
    ns.update(helpers_mod.HELPER_EXPORTS)
    ns.update({
        "WorldModelLibrary": wmlib_mod.WorldModelLibrary,
        "TransitionRecord": wmlib_mod.TransitionRecord,
    })
    try:
        for src in extra_sources:
            if src:
                exec(compile(src, "<retained>", "exec"), ns)
        if project_source:
            exec(compile(project_source, "<project>", "exec"), ns)
        if simulate_source:
            exec(compile(simulate_source, "<simulate>", "exec"), ns)
        if is_goal_source:
            exec(compile(is_goal_source, "<is_goal>", "exec"), ns)
    except Exception as e:
        return ns, f"{type(e).__name__}: {e}"
    return ns, None


def check_level_consolidated(library, ledger, kept_functions, level):
    """Consolidation status for `level` using retained top-level defs.

    kept_functions: dict {name: source} from ToolAgent._kept_functions.
    Automatically detects whether `project` is retained and passes it to
    evaluate_model (falling back to Grid mode if projected evaluation fails
    because `simulate` expects a SimFrame).
    """
    kept = kept_functions or {}
    sim_src = kept.get("simulate")
    goal_src = kept.get("is_goal")
    proj_src = kept.get("project")
    win_rec = library.win_transition(level)
    base = {
        "consolidated": False,
        "has_simulate": bool(sim_src),
        "has_is_goal": bool(goal_src),
        "has_project": bool(proj_src),
        "win_seq_id": win_rec.seq_id if win_rec is not None else -1,
        "win_action": win_rec.action if win_rec is not None else "",
        "accuracy": None,
        "reason": "",
    }
    if not sim_src or not goal_src:
        base["reason"] = "simulate and/or is_goal not retained yet"
        return base

    extras = [
        src for name, src in kept.items()
        if name not in ("simulate", "is_goal", "project")
    ]
    ns, err = build_model_namespace(
        sim_src, goal_src, project_source=proj_src, extra_sources=extras, library=library
    )
    if err is not None:
        base["reason"] = f"model sources failed to exec: {err}"
        return base
    simulate = ns.get("simulate")
    is_goal = ns.get("is_goal")
    project = ns.get("project") if proj_src else None
    if not callable(simulate) or not callable(is_goal):
        base["reason"] = "simulate/is_goal are not callable after exec"
        return base

    verdict = evaluate_model(
        library, ledger, simulate, is_goal,
        level=level, project=project if callable(project) else None,
    )
    # If `project` was retained from an earlier experiment but `simulate` is
    # written for SimFrame, also try Grid mode so a valid Grid model passes.
    if not verdict.get("ok") and callable(project):
        verdict_grid = evaluate_model(library, ledger, simulate, is_goal, level=level, project=None)
        if verdict_grid.get("ok") or (
            (verdict_grid.get("simulation_accuracy") or 0.0)
            > (verdict.get("simulation_accuracy") or 0.0)
        ):
            verdict = verdict_grid

    base["accuracy"] = verdict.get("simulation_accuracy")
    base["consolidated"] = bool(verdict.get("ok"))
    base["reason"] = "PASS" if verdict.get("ok") else (
        "check_sim not PASS yet (accuracy="
        + (f"{verdict['simulation_accuracy']:.2f}" if verdict.get("simulation_accuracy") is not None else "n/a")
        + ")"
    )
    return base
