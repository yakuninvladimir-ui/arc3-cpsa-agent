"""CPSA 2.0 acceptance suite + extended & v2.0 regression tests.

Run:  python3 tests/test_cpsa_acceptance.py
"""
import os
import sys
import shutil
import tempfile

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)
_agent = os.path.join(_root, "inference", "agent")
if os.path.isdir(os.path.join(_agent, "cpsa")) and _agent not in sys.path:
    sys.path.insert(0, _agent)

import numpy as np

from cpsa.wmlib import WorldModelLibrary, diff_text, resolve_color, color_char
from cpsa.world_model import (
    SimFrame, FalsifiedGoalLedger, bind_world_model_api, evaluate_model,
    cells, bbox, colors, show_window, candidate_clicks, copy_state,
    expect_moved, expect_region_changed, expect_diff_count, expect_cells,
)
from cpsa.helpers import bg_color, find_color, first_of, objects_of, move_object
from cpsa.invariant_store import GameInvariantStore, ReasoningLifecycleManager
from cpsa.journal import CpsaJournalWriter, load_library, save_ledger, load_ledger
from cpsa.guards import CpsaGuards
from cpsa.consolidation import check_level_consolidated
from cpsa import sandbox_bridge


def _make_env():
    """Synthetic 64x64 game: player color 9 at (10,10), wall color 5 at
    (10,12), goal color 4 at (10,13). Row 0 = ticking HUD timer."""
    board = np.zeros((64, 64), dtype=np.int8)
    board[10, 10] = 9
    board[10, 12] = 5
    board[10, 13] = 4
    board[0, :50] = 2
    env = {"board": board, "timer_col": 49, "level": 1}
    wmlib = WorldModelLibrary()
    ledger = FalsifiedGoalLedger()
    store = GameInvariantStore()
    falsified_events = []
    wmlib.on_level_start(1, board)

    def step_env(act):
        board = env["board"]
        prev = board.copy()
        nxt = board.copy()
        if env["timer_col"] >= 0:
            nxt[0, env["timer_col"]] = 0
            env["timer_col"] -= 1
        ys, xs = np.nonzero(board == 9)
        r, c = int(ys[0]), int(xs[0])
        dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1),
                  "RIGHT": (0, 1)}.get(act, (0, 0))
        nr, nc = r + dr, c + dc
        if 0 < nr < 64 and 0 <= nc < 64 and board[nr, nc] != 5:
            nxt[r, c] = 0
            nxt[nr, nc] = 9
        lvl_done = (nr == 10 and nc == 13)
        if lvl_done:
            pre_swap = nxt.copy()
            lvl2 = np.zeros((64, 64), dtype=np.int8)
            lvl2[20, 20] = 9
            env["board"] = lvl2
            rec = wmlib.record_transition(
                action=act, before=prev, after=lvl2.copy(),
                anim_frames=[prev.copy(), pre_swap, lvl2.copy()],
                level_completed=True, game_over=False, level_override=env["level"],
            )
            env["level"] = 2
            wmlib.on_level_start(2, lvl2)
            return rec
        env["board"] = nxt
        return wmlib.record_transition(
            action=act, before=prev, after=nxt.copy(), anim_frames=[nxt.copy()],
            level_completed=False, game_over=False, level_override=env["level"],
        )

    api = bind_world_model_api(
        wmlib=wmlib, falsified_ledger=ledger, step_single_action_fn=step_env,
        get_current_grid_fn=lambda: env["board"],
        get_current_level_fn=lambda: env["level"],
        notify_hypothesis_falsified_fn=lambda msg: falsified_events.append(msg),
    )
    return env, wmlib, ledger, store, falsified_events, api


def _sim_good():
    def sim_good(frame, act):
        f = frame.copy()
        f.state["steps"] = f.state.get("steps", 0) + 1
        pts = f.find(9)
        if pts:
            r, c = pts[0]
            dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1),
                      "RIGHT": (0, 1)}.get(act, (0, 0))
            nr, nc = r + dr, c + dc
            if 0 < nr < 64 and 0 <= nc < 64 and f[nr, nc] != 5:
                f[r, c] = 0
                f[nr, nc] = 9
        f[60, 60] = None
        return f
    return sim_good


def run_acceptance_suite():
    env, wmlib, ledger, store, falsified_events, api = _make_env()
    board_ref = lambda: env["board"]

    # TEST 1: tail abort on expectation mismatch
    res1 = api["act_expect"](["RIGHT", "RIGHT", "RIGHT"], expect_color_moved=9)
    assert res1["status"] == "EXPECTATION_MISMATCH", res1
    assert res1["executed"] == ["RIGHT", "RIGHT"], res1["executed"]
    assert res1["remaining"] == ["RIGHT"]
    assert len(falsified_events) == 1

    # TEST 2: dynamic HUD mask on row 0
    res_down = api["act_expect"](["DOWN"], expect=expect_diff_count(1, 10))
    assert res_down["status"] == "COMPLETED", res_down
    hud = wmlib.compute_hud_mask(1)
    assert np.all(hud[0, :]), "row 0 must be masked"
    assert not np.any(hud[1:63, 1:63]), "interior must not be masked"

    # TEST 3: check_sim PASS with state threading + legal None abstention
    sim_good = _sim_good()
    chk = api["check_sim"](sim_good, level=1)
    assert chk["ok"] is True, chk["report"]
    assert chk["v4_counts"]["FOLLOW"] >= 2 and chk["v4_counts"]["OMIT"] >= 1

    # TEST 4: masking real changes with None must fail
    def sim_cheat(frame, act):
        f = frame.copy()
        for r in range(9, 13):
            for c in range(9, 14):
                f[r, c] = None
        return f
    chk_cheat = api["check_sim"](sim_cheat, level=1)
    assert chk_cheat["ok"] is False

    # TEST 5: falsified goal blacklisted, search refuses at 0 cost
    false_goal = lambda f: (11, 12) in f.find(9)
    plan_false = api["search"](board_ref(), sim_good, false_goal)
    assert plan_false == ["RIGHT"], plan_false
    exec_false = api["act_checked"](plan_false, sim_good, false_goal)
    assert exec_false["status"] == "GOAL_FALSIFIED", exec_false
    try:
        api["search"](board_ref(), sim_good, false_goal)
        raise AssertionError("search() must reject a falsified goal!")
    except RuntimeError as e:
        assert "REJECTED BY JUDGE" in str(e)

    # TEST 6: true goal found and executed
    true_goal = lambda f: (10, 13) in f.find(9)
    plan_true = api["search"](board_ref(), sim_good, true_goal)
    assert plan_true == ["RIGHT", "UP"], plan_true
    exec_true = api["act_checked"](plan_true, sim_good, true_goal)
    assert exec_true["status"] == "LEVEL_COMPLETED", exec_true
    assert env["level"] == 2

    # TEST 7: post-win retrodiction of level 1 incl. win transition
    chk_post = api["check_sim"](sim_good, true_goal, level=1)
    assert chk_post["ok"] is True and chk_post["goal_status"] == "CONFIRMED", chk_post["report"]

    # TEST 8: invariant store + reasoning lifecycle
    store.update_invariants(
        mechanics={"avatar": "[checked] color 9 moves 1 step orthogonally",
                   "wall": "[checked] color 5 blocks movement"},
        win_condition="[checked] Reach target cell (color 4)",
        level=1, level_delta="Initial level with 1 wall block",
    )
    dumped = store.to_dict()
    store2 = GameInvariantStore()
    store2.load_dict(dumped)
    rendered = store2.render_for_prompt(current_level=2)
    assert "color 9 moves 1 step orthogonally" in rendered
    assert "Reach target cell (color 4)" in rendered
    assert "[checked]" in rendered

    rmgr = ReasoningLifecycleManager(keep_last_k=2)
    msgs = [
        {"role": "assistant", "content": "c1", "reasoning_content": "r1"},
        {"role": "assistant", "content": "c2", "reasoning_content": "r2"},
        {"role": "assistant", "content": "c3", "reasoning_content": "r3"},
    ]
    rmgr.prune_stale_reasoning(msgs)
    assert "reasoning_content" not in msgs[0] and "reasoning_content" in msgs[1]
    n, limited = rmgr.amputate_all_reasoning_on_falsification(msgs, "EXPECTATION_MISMATCH")
    assert not limited and n == 2
    assert all("reasoning_content" not in m for m in msgs)

    print("ALL 8 CPSA ACCEPTANCE TESTS PASSED!")
    return env, wmlib, ledger, api


def test_diff_text():
    a = np.zeros((64, 64), dtype=np.int8)
    b = a.copy()
    b[12, 4:10] = 9
    txt = diff_text(a, b)
    assert "6 cells changed" in txt and "0->9" in txt and "row 12" in txt and "cols 4-9" in txt, txt
    assert diff_text(a, a) == "no cells changed"
    print("diff_text OK")


def test_utils():
    g = np.zeros((64, 64), dtype=np.int8)
    g[20, 20] = 9
    g[5, 5] = 4
    assert cells(g, 9) == [(20, 20)]
    assert cells(g, "b") == [(20, 20)]  # ARC char 'b' == 9
    assert bbox("b", g) == (20, 20, 20, 20)
    assert colors(g)[9] == 1 and colors(g)[0] == 64 * 64 - 2
    w = show_window(g, 19, 19, 21, 21)
    assert "9" in w
    try:
        show_window(g, 0, 0, 63, 63)
        raise AssertionError("full-board show_window must be refused")
    except RuntimeError as e:
        assert "32x32" in str(e)
    f = SimFrame(g)
    clicks = candidate_clicks(f)
    assert "MOUSE(row=20, col=20)" in clicks and "MOUSE(row=5, col=5)" in clicks
    print("utils OK")


def test_expect_cells():
    prev = SimFrame(np.zeros((64, 64), dtype=np.int8))
    cur = prev.copy()
    cur[3, 4] = 9
    ok, why = expect_cells([(3, 4, "b")])(prev, cur, "RIGHT", [(3, 4, 0, 9)])
    assert ok, why
    bad, _ = expect_cells([(3, 4, 8)])(prev, cur, "RIGHT", [(3, 4, 0, 9)])
    assert not bad
    extra, why3 = expect_cells([])(prev, cur, "RIGHT", [(3, 4, 0, 9)])
    assert not extra and "extra" in why3
    print("expect_cells OK")


def test_journal_roundtrip():
    tmp = tempfile.mkdtemp(prefix="cpsa_journal_")
    try:
        w = CpsaJournalWriter(tmp)
        g0 = np.zeros((64, 64), dtype=np.int8)
        g0[1, 1] = 9
        w.on_level_start(1, g0)
        g1 = g0.copy()
        g1[1, 1] = 0
        g1[1, 2] = 9
        w.record_transition("RIGHT", g0, g1, anim_frames=[g1])
        w.on_attempt_reset()
        w.record_transition("LEFT", g1, g0, level_completed=False, game_over=True)
        w.close()
        lib = load_library(tmp)
        recs = lib.transitions(level=1)
        assert len(recs) == 2 and recs[0].action == "RIGHT"
        assert recs[1].attempt == 2 and recs[1].game_over
        led = FalsifiedGoalLedger()
        led.record_false_positive_goal(1, g1, {"k": 3}, 7, lambda f: True)
        save_ledger(tmp, led)
        led2 = load_ledger(tmp)
        ok, err = led2.validate_goal_candidate(1, lambda f: True)
        assert not ok and "REJECTED BY JUDGE" in err
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("journal roundtrip OK")


def test_guards_corridor_and_cycle():
    def h(rows):
        return hash(tuple(tuple(r) for r in rows))
    g = CpsaGuards(h)
    grid_a = [[0] * 8 for _ in range(8)]
    blocked = 0
    for i in range(30):
        rows = [[(i * 13 + r * 8 + c) for c in range(8)] for r in range(8)]
        v = g.check(rows, "RIGHT", None, 1)
        g.note_executed_action(1, "RIGHT")
        if v:
            blocked += 1
    assert blocked == 0
    g2 = CpsaGuards(h)
    res = [g2.check(grid_a, "RIGHT", None, 2) for _ in range(3)]
    assert all(r is None for r in res)
    r4 = g2.check(grid_a, "RIGHT", None, 2)
    assert r4 is not None and r4[0] == "cpsa_cycle_repeat"
    r5 = g2.check(grid_a, "RIGHT", None, 2)
    assert r5 is None
    print("guards corridor/cycle OK")


def test_guards_clicks_and_escalation():
    def h(rows):
        return hash(tuple(tuple(r) for r in rows))
    g = CpsaGuards(h)
    for i in range(25):
        grid = [[0] * 8 for _ in range(8)]
        grid[i % 8][(i * 3) % 8] = 1 + (i % 15)
        v = g.check(grid, "MOUSE", {"row": 10, "col": 12}, 1)
        g.note_executed_action(1, "MOUSE", {"row": 10, "col": 12})
        assert v is None
    grid = [[0] * 8 for _ in range(8)]
    v = g.check(grid, "MOUSE", {"row": 10, "col": 12}, 1)
    assert v is not None and v[0] == "cpsa_click_repeat"
    assert g.check(grid, "MOUSE", {"row": 10, "col": 12}, 1) is None
    g3 = CpsaGuards(h)
    for _ in range(125):
        g3.note_executed_action(3, "RIGHT")
    assert "ESCALATION T1" in g3.directive(3)
    for _ in range(130):
        g3.note_executed_action(3, "RIGHT")
    assert "ESCALATION T2" in g3.directive(3)
    for _ in range(150):
        g3.note_executed_action(3, "RIGHT")
    assert "SCORE FLOOR" in g3.directive(3)
    print("guards clicks/escalation OK")


def test_advisory_tier():
    env, wmlib, ledger, _, _, api = _make_env()
    sim_good = _sim_good()

    def sim_flawed(frame, act):
        f = sim_good(frame, act)
        if f.state.get("steps") == 5:
            f.grid[55, 55] = 7
        return f

    wander = ["DOWN", "RIGHT", "UP", "LEFT"] * 3
    r = api["act_expect"](wander, expect=None)
    assert r["status"] == "COMPLETED"
    res = evaluate_model(wmlib, ledger, sim_flawed, None, level=1)
    assert res["ok"] is False and res["advisory_ok"] is True
    true_goal = lambda f: (10, 13) in f.find(9)
    plan = api["search"](env["board"], sim_flawed, true_goal)
    assert plan is not None and len(plan) > 0
    print("advisory tier OK")


def test_bridge_and_consolidation():
    tmp = tempfile.mkdtemp(prefix="cpsa_bridge_")
    try:
        board = np.zeros((64, 64), dtype=np.int8)
        board[10, 10] = 9
        board[10, 13] = 4
        writer = CpsaJournalWriter(tmp)
        writer.on_level_start(1, board)
        state = {"board": board, "level": 1}

        def fake_step(payload):
            act = payload[0] if isinstance(payload, list) else payload
            name = act if isinstance(act, str) else act.get("action")
            prev = state["board"].copy()
            nxt = prev.copy()
            ys, xs = np.nonzero(prev == 9)
            r, c = int(ys[0]), int(xs[0])
            dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1),
                      "RIGHT": (0, 1)}.get(str(name).upper(), (0, 0))
            nr, nc = r + dr, c + dc
            done = (nr == 10 and nc == 13)
            if 0 <= nr < 64 and 0 <= nc < 64:
                nxt[r, c] = 0
                nxt[nr, nc] = 9
            writer.record_transition(
                str(name).upper(), prev, nxt,
                level_completed=done, level_override=state["level"],
            )
            if done:
                state["level"] = 2
                writer.on_level_start(2, nxt)
            state["board"] = nxt
            return {"executed": True, "level_completed": done, "game_over": False,
                    "run_complete": False, "level": state["level"]}

        exports, final_fields = sandbox_bridge.build({
            "enabled": True, "journal_dir": tmp, "consolidation_readonly": False,
            "step": fake_step,
            "grid": lambda: state["board"],
            "level": lambda: state["level"],
            "anim_frames": lambda: [],
        })
        r = exports["act_expect"](
            ["RIGHT", "RIGHT", "RIGHT"],
            expect=exports["expect_moved"]("b", dc=1),
        )
        assert r["status"] in ("COMPLETED", "LEVEL_COMPLETED"), r
        exports["save_invariants"](
            mechanics={"avatar": "[checked] color 9"},
            win_condition="[assumed] reach color 4",
        )
        ff = final_fields()
        assert ff["invariant_store_state"]["confirmed_mechanics"]["avatar"].startswith("[checked]")

        # Consolidation with Boundary Projection (project + simulate + is_goal)
        proj_src = "def project(frame):\n    return first_of(frame, 'b')\n"
        sim_src = (
            "def simulate(st, action):\n"
            "    r, c = st\n"
            "    d = {'UP': (-1,0), 'DOWN': (1,0), 'LEFT': (0,-1), 'RIGHT': (0,1)}.get(action, (0,0))\n"
            "    return (r + d[0], c + d[1])\n"
        )
        goal_src = "def is_goal(st):\n    return st == (10, 13)\n"
        lib = load_library(tmp)
        status = check_level_consolidated(
            lib, load_ledger(tmp),
            {"project": proj_src, "simulate": sim_src, "is_goal": goal_src},
            1,
        )
        assert status["consolidated"] is True, status
        writer.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("bridge + consolidation (with project) OK")


def test_hud_report():
    env, wmlib, ledger, store, events, api = _make_env()
    api["act_expect"](["RIGHT"], expect=None)
    api["act_expect"](["DOWN"], expect=None)
    rep = wmlib.hud_report(1)
    assert "rows=[0]" in rep and "actions left" in rep, rep
    print("hud_report OK")


def test_redundant_reset_no_typeerror():
    tmp = tempfile.mkdtemp(prefix="cpsa_reset_")
    try:
        board = np.zeros((64, 64), dtype=np.int8)
        board[10, 10] = 9
        state = {"board": board, "level": 1, "env_steps": 0}

        def fake_step(payload):
            state["env_steps"] += 1
            return {"executed": True, "level_completed": False,
                    "game_over": False, "run_complete": False, "level": 1}

        exports, _ = sandbox_bridge.build({
            "enabled": True, "journal_dir": tmp, "consolidation_readonly": False,
            "step": fake_step,
            "grid": lambda: state["board"],
            "level": lambda: state["level"],
            "anim_frames": lambda: [],
        })
        exports["wmlib"].on_level_start(1, board)
        res = exports["act_expect"](["RESET", "RIGHT"])
        assert res["status"] == "COMPLETED" and state["env_steps"] == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("redundant RESET guard OK")


def test_frameview_and_arc_color_chars():
    """v2.0 regression: FrameView exposes ._grid (no .grid), background may be
    non-zero (e.g. 5='B' black), and functions accept ARC color characters."""
    class DummyFrameView:
        def __init__(self, rows):
            self._grid = rows

    raw = [[5] * 64 for _ in range(64)]
    raw[10][10] = 8  # 'R'
    raw[10][11] = 8
    fv = DummyFrameView(raw)
    assert bg_color(fv) == 5
    assert cells(fv, "R") == [(10, 10), (10, 11)]
    objs = objects_of(fv)
    assert len(objs) == 1 and objs[0]["char"] == "R" and objs[0]["pixels"] == 2
    sf = SimFrame(fv)
    assert move_object(sf, "R", 1, 0)
    assert int(sf.grid[10, 10]) == 5 and int(sf.grid[11, 10]) == 8
    clicks = candidate_clicks(fv)
    assert clicks == ["MOUSE(row=10, col=10)"]
    print("FrameView._grid + ARC_COLOR_CHARS + non-zero bg OK")


def test_boundary_projection_quotient_mode():
    """v2.0: Boundary Projection pi_K with kernel inclusion check."""
    env, wmlib, ledger, store, events, api = _make_env()
    api["act_expect"](["DOWN", "RIGHT"], expect=None)

    # Degenerate projector (violates ker(project) <= ker(env)) must be rejected
    bad_proj = lambda f: 0
    bad_sim = lambda s, a: 0
    bad_goal = lambda s: False
    v_bad = api["check_sim"](bad_sim, bad_goal, project=bad_proj)
    assert v_bad["ok"] is False and "KERNEL INCLUSION VIOLATION" in v_bad["report"]

    # Valid projector on avatar coordinates
    proj = lambda f: first_of(f, "b")

    def sim_proj(st, act):
        r, c = st
        dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1),
                  "RIGHT": (0, 1)}.get(act, (0, 0))
        nr, nc = r + dr, c + dc
        if (nr, nc) == (10, 12):
            return (r, c)
        return (nr, nc)

    goal_proj = lambda st: st == (10, 13)
    v_ok = api["check_sim"](sim_proj, goal_proj, project=proj)
    assert v_ok["ok"] is True, v_ok["report"]
    plan = api["search"](env["board"], sim_proj, goal_proj, project=proj)
    assert plan == ["RIGHT", "RIGHT", "UP"], plan
    res = api["act_checked"](plan, sim_proj, goal_proj, project=proj)
    assert res["status"] == "LEVEL_COMPLETED"
    print("Boundary Projection quotient mode OK")


def test_reasoning_both_keys():
    """v2.0 regression: tool_agent stores reasoning under 'reasoning' by default."""
    rmgr = ReasoningLifecycleManager(keep_last_k=1)
    msgs = [
        {"role": "assistant", "content": "a1", "reasoning": "think1"},
        {"role": "assistant", "content": "a2", "reasoning": "think2"},
    ]
    stripped = rmgr.prune_stale_reasoning(msgs)
    assert stripped == 1 and "reasoning" not in msgs[0] and "reasoning" in msgs[1]
    amp, _ = rmgr.amputate_all_reasoning_on_falsification(msgs, "SIM_DIVERGENCE")
    assert amp == 1 and "reasoning" not in msgs[1]
    print("reasoning both keys ('reasoning' & 'reasoning_content') OK")


def test_pre_level_swap_grid_no_false_before():
    """v2.0 regression: when collapse_frames produces [before, after_new_level]
    (len == 2), pre_level_swap_grid must be None, not `before`."""
    lib = WorldModelLibrary()
    b = np.zeros((64, 64), dtype=np.int8)
    b[1, 1] = 9
    a = np.zeros((64, 64), dtype=np.int8)
    a[30, 30] = 9
    rec = lib.record_transition(
        "RIGHT", b, a, anim_frames=[b.copy(), a.copy()], level_completed=True
    )
    assert rec.pre_level_swap_grid is None
    mid = b.copy()
    mid[1, 2] = 9
    rec2 = lib.record_transition(
        "RIGHT", b, a, anim_frames=[b.copy(), mid, a.copy()], level_completed=True
    )
    assert rec2.pre_level_swap_grid is not None and np.array_equal(rec2.pre_level_swap_grid, mid)
    print("pre_level_swap_grid guard OK")


def test_vacuous_truth_and_undecided_v4():
    """Axiom 4.1 MBEL:
    1) Identity simulator on OMIT-only history (FOLLOW == 0) cannot get PASS.
    2) Constant-False `is_goal` stub (e.g. from SIM_SKELETON) is rejected.
    3) Masking active changes with None (<=15%) is counted as UNDECIDED (U), not NULL.
    """
    env, wmlib, ledger, store, events, api = _make_env()

    # Move avatar to (10, 11) and hit wall at (10, 12) on level 1;
    # then inspect a fresh level where ONLY a blocked (OMIT) move was recorded:
    wmlib.on_level_start(99, env["board"])
    wmlib.record_transition(
        action="UP", before=env["board"], after=env["board"],
        anim_frames=[env["board"]], level_completed=False, game_over=False,
        level_override=99,
    )
    v_noop = evaluate_model(wmlib, ledger, lambda f, a: f.copy(), level=99)
    assert v_noop["ok"] is False and v_noop["advisory_ok"] is False
    assert v_noop["v4_counts"]["OMIT"] == 1 and v_noop["v4_counts"]["FOLLOW"] == 0
    assert "VACUOUS MODEL VIOLATION" in v_noop["report"]

    # Switch back to level 1 and record active transitions
    wmlib._current_level = 1
    api["act_expect"](["RIGHT", "DOWN"], expect=None)
    sim_good = _sim_good()

    # Constant-False goal stub from SIM_SKELETON must be rejected by Axiom 4.1 / NonTriv(y)
    def stub_goal(frame):
        return False

    v_stub = api["check_sim"](sim_good, stub_goal, level=1)
    assert v_stub["ok"] is False and v_stub["goal_status"] == "CONTRADICTED"
    assert "VACUOUS GOAL VIOLATION" in v_stub["report"]

    # Masking an active change with None (within 15%) increments UNDECIDED (U)
    def sim_abstain_active(frame, act):
        f = frame.copy()
        for r in range(9, 13):
            for c in range(9, 14):
                f[r, c] = None
        return f

    v_abs = api["check_sim"](sim_abstain_active, level=1)
    assert v_abs["ok"] is False
    assert v_abs["v4_counts"]["UNDECIDED"] == 2 and v_abs["v4_counts"]["NULL"] == 0
    assert "U=2" in v_abs["report"]
    print("Axiom 4.1 vacuous truth guard + V4 UNDECIDED accounting OK")




def test_cpsa_v2_1_regressions():
    """Verify all CPSA 2.1 fixes:
      1. Turn-0 cross-level search() & check_sim() after Level 1 consolidation (Bug #15)
      2. FalsifiedGoalLedger JSON roundtrip preserving Python tuples (Bug #13)
      3. Boundary Projection (pi_K) latent state threading with '_' keys (Bug #1)
      4. Active accuracy gate preventing OMIT wall-bump inflation in ADVISORY-OK (Bug #2)
      5. Winning step requiring goal_on_pred (no OR bypass via pre_level_swap_grid) (Bug #14)
      6. Hazard pruning (SimFrame.dead / state['dead']) in search() & act_checked(None) (Bug #3)
      7. move_object pushing multi-cell (2x2) boxes (Bug #10)
      8. candidate_clicks prioritizing interactive sizes (1..128 px) & snapping off-component centroids (Bug #11)
      9. HUD mask cross-level inheritance, identical-action detection, and player-walk false-positive guard (Bug #12)
      10. CpsaHost RESET filtering, GAME_OVER auto-reset, and HUD-masked cycle hash (Bugs #4, #5, #17)
      11. sandbox_bridge wrapped action() enforcing CONSOLIDATION_READONLY & wmlib sync (Bugs #7, #16)
    """
    import tempfile
    from inference.agent.cpsa import journal as cpsa_journal
    from inference.agent.cpsa_host import CpsaHost
    from inference.agent.cpsa.sandbox_bridge import build as build_bridge

    # (1) & (9a): Turn-0 cross-level search() and HUD inheritance on Level 2
    env, wmlib, ledger, store, events, api = _make_env()
    sim_good = _sim_good()
    goal_good = lambda f: (10, 13) in f.find(9)
    api["act_expect"](["RIGHT", "DOWN"], expect=None)
    # Complete level 1 via act_checked
    plan_l1 = api["search"](env["board"], sim_good, goal_good)
    res_l1 = api["act_checked"](plan_l1, sim_good, goal_good)
    assert res_l1["status"] == "LEVEL_COMPLETED"
    # Advance to Level 2 with 0 transitions on Level 2
    l2_board = np.zeros((64, 64), dtype=np.int16)
    l2_board[0, :15] = 2
    l2_board[10, 10] = 9
    l2_board[10, 12] = 5
    l2_board[10, 13] = 4
    env["board"] = l2_board.copy()
    env["level"] = 2
    wmlib.on_level_start(2, l2_board)
    # HUD mask on Level 2 Turn 0 is inherited from Level 1!
    hud_l2 = wmlib.compute_hud_mask(2)
    assert np.all(hud_l2[0, :]), "Level 2 Turn 0 should inherit HUD mask from Level 1"
    # Turn-0 search() on Level 2 succeeds using consolidated Level 1 certification!
    plan_l2 = api["search"](env["board"], sim_good, goal_good)
    assert plan_l2 == ["UP", "RIGHT", "RIGHT", "RIGHT", "DOWN"], f"Unexpected Turn-0 plan: {plan_l2}"

    # (2) FalsifiedGoalLedger tuple preservation across JSON serialization
    tmp_dir = tempfile.mkdtemp(prefix="cpsa_v21_ledger_")
    try:
        led = FalsifiedGoalLedger()
        g0 = np.zeros((64, 64), dtype=np.int8)
        led.record_false_positive_goal(1, g0, (10, 13), seq_id=7, is_goal_fn=lambda st: st == (10, 13))
        cpsa_journal.save_ledger(tmp_dir, led)
        led_reloaded = cpsa_journal.load_ledger(tmp_dir)
        ok_tup, err_tup = led_reloaded.validate_goal_candidate(1, lambda st: st == (10, 13), project=lambda f: f.state)
        assert not ok_tup and "REJECTED BY JUDGE" in err_tup, "Reloaded tuple state must still reject tuple equality goal"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # (3) Boundary Projection (pi_K) latent state threading with '_' keys
    env3, wmlib3, ledger3, store3, _, api3 = _make_env()
    api3["act_expect"](["DOWN", "RIGHT"], expect=None)
    proj_latent = lambda f: {"pos": first_of(f, "b")}
    def sim_latent(st, act):
        s = dict(st)
        r, c = s["pos"]
        dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}.get(act, (0, 0))
        nr, nc = r + dr, c + dc
        if (nr, nc) == (10, 12):
            nr, nc = r, c
        s["pos"] = (nr, nc)
        s["_steps"] = int(s.get("_steps", 0)) + 1
        return s
    goal_latent = lambda st: st.get("pos") == (10, 13) and int(st.get("_steps", 0)) >= 3
    v_lat = api3["check_sim"](sim_latent, goal_latent, project=proj_latent)
    assert v_lat["ok"] is True, v_lat["report"]
    plan_lat = api3["search"](env3["board"], sim_latent, goal_latent, project=proj_latent)
    assert plan_lat == ["RIGHT", "RIGHT", "UP"], plan_lat

    # (4) Active accuracy gate prevents OMIT no-op inflation in ADVISORY-OK
    lib_omit = WorldModelLibrary()
    b0 = np.zeros((64, 64), dtype=np.int8)
    b0[10, 10] = 9
    b1 = b0.copy(); b1[10, 11] = 9; b1[10, 10] = 0
    b2 = b1.copy(); b2[11, 11] = 9; b2[10, 11] = 0
    lib_omit.on_level_start(1, b0)
    # 1 FOLLOW move + 1 NULL active move + 10 OMIT wall bumps -> raw acc = 11/12 = 91.7%, active_acc = 1/2 = 50%
    lib_omit.record_transition("RIGHT", b0, b1)
    lib_omit.record_transition("DOWN", b1, b2)
    for _ in range(10):
        lib_omit.record_transition("UP", b2, b2)
    def sim_half_broken(frame, act):
        f = frame.copy()
        if act == "RIGHT":
            move_object(f, 9, 0, 1)
        return f
    v_omit = evaluate_model(lib_omit, FalsifiedGoalLedger(), sim_half_broken, level=1)
    assert v_omit["ok"] is False and v_omit["advisory_ok"] is False, "50% active accuracy must NOT pass ADVISORY-OK"

    # (5) Winning step requires goal_on_pred (no OR bypass via pre_level_swap_grid)
    lib_win = WorldModelLibrary()
    wb = np.zeros((64, 64), dtype=np.int8); wb[10, 12] = 9
    wmid = np.zeros((64, 64), dtype=np.int8); wmid[9, 12] = 9
    wa = np.zeros((64, 64), dtype=np.int8); wa[30, 30] = 9
    lib_win.on_level_start(1, wb)
    lib_win.record_transition("UP", wb, wa, anim_frames=[wb, wmid, wa], level_completed=True)
    # Simulator that is a no-op on UP (does not reach (9,12)) must fail winning transition check
    v_win_bad = evaluate_model(
        lib_win, FalsifiedGoalLedger(),
        simulate=lambda f, a: f.copy(),
        is_goal=lambda f: (9, 12) in f.find(9),
        level=1,
    )
    assert v_win_bad["ok"] is False and v_win_bad["goal_status"] == "CONTRADICTED"

    # (6) Hazard pruning (SimFrame.dead) in search() and act_checked(None)
    def sim_with_hazard(frame, act):
        f = sim_good(frame, act)
        if (11, 10) in f.find(9):
            f.dead = True
        return f
    env6, wmlib6, ledger6, store6, _, api6 = _make_env()
    api6["act_expect"](["RIGHT", "LEFT"], expect=None)
    plan_safe = api6["search"](env6["board"], sim_with_hazard, goal_good)
    assert plan_safe is not None and plan_safe[0] != "DOWN"
    empty_res = api6["act_checked"](None, sim_with_hazard, goal_good)
    assert empty_res["status"] == "EMPTY_PLAN" and empty_res["executed"] == []

    # (7) move_object pushing a 2x2 box
    sf_box = SimFrame(np.zeros((64, 64), dtype=np.int16))
    sf_box.grid[10, 10] = 9
    sf_box.grid[10, 11:13] = 11  # 1x2 horizontal box of color 11
    sf_box.grid[11, 11:13] = 11  # 2x2 box of color 11
    assert move_object(sf_box, 9, 0, 1, push=11) is True
    assert int(sf_box.grid[10, 11]) == 9
    assert np.all(sf_box.grid[10:12, 12:14] == 11)

    # (8) candidate_clicks interactive-size priority & L-shape centroid snap
    g_click = np.zeros((64, 64), dtype=np.int16)
    g_click[0:4, 0:64] = 5  # 256-px border wall
    g_click[20, 20:24] = 8  # 4-px button
    clicks = candidate_clicks(g_click)
    assert clicks[0].startswith("MOUSE(row=20,"), f"Interactive 4-px button must precede 256-px wall, got {clicks}"

    # (9b & 9c) HUD detection on repeated ["DOWN", "DOWN"] and NO false positive on player walking along row 1
    lib_hud = WorldModelLibrary()
    h0 = np.zeros((64, 64), dtype=np.int8); h0[0, :10] = 2; h0[10, 10] = 9
    h1 = h0.copy(); h1[0, 9] = 0; h1[11, 10] = 9; h1[10, 10] = 0
    h2 = h1.copy(); h2[0, 8] = 0; h2[12, 10] = 9; h2[11, 10] = 0
    lib_hud.on_level_start(1, h0)
    lib_hud.record_transition("DOWN", h0, h1)
    lib_hud.record_transition("DOWN", h1, h2)
    assert np.all(lib_hud.compute_hud_mask(1)[0, :]), "Row 0 shrinking 1px/step must be detected even on ['DOWN','DOWN']"

    lib_walk = WorldModelLibrary()
    w0 = np.zeros((64, 64), dtype=np.int8); w0[1, 10] = 9
    w1 = np.zeros((64, 64), dtype=np.int8); w1[1, 11] = 9
    w2 = np.zeros((64, 64), dtype=np.int8); w2[2, 11] = 9
    lib_walk.on_level_start(1, w0)
    lib_walk.record_transition("RIGHT", w0, w1)
    lib_walk.record_transition("DOWN", w1, w2)
    cur = w2
    for step_i in range(6):
        nxt = np.zeros((64, 64), dtype=np.int8); nxt[3 + step_i, 11] = 9
        lib_walk.record_transition("DOWN", cur, nxt)
        cur = nxt
    assert not np.any(lib_walk.compute_hud_mask(1)[1, :]), "Player walking along row 1 for 2/8 steps must NOT be masked as HUD"

    # (10) CpsaHost RESET filtering, GAME_OVER auto-reset, and HUD-masked cycle hash
    tmp_host = tempfile.mkdtemp(prefix="cpsa_v21_host_")
    try:
        host = CpsaHost(lambda rows: hash(tuple(tuple(r) for r in rows)))
        host.attach(os.path.join(tmp_host, "state.json"))
        host.on_transition({
            "level_before": 1, "level_after": 1, "action_display": "DOWN",
            "before": h0.tolist(), "after": h1.tolist(), "anim_chain": [],
            "level_completed": False, "game_over": False, "automatic": False, "action_num": 1,
        })
        host.on_transition({
            "level_before": 1, "level_after": 1, "action_display": "RESET",
            "before": h1.tolist(), "after": h0.tolist(), "anim_chain": [],
            "level_completed": False, "game_over": False, "automatic": False, "action_num": 2,
        })
        # RESET transition must not appear in wmlib.transitions(include_reset=False)
        recs_active = host.journal.library.transitions(level=1)
        assert len(recs_active) == 1 and recs_active[0].action == "DOWN"
        assert host.journal.library.current_attempt == 2
        # GAME_OVER transition must automatically advance attempt even if auto-reset happens outside analyze()
        host.on_transition({
            "level_before": 1, "level_after": 1, "action_display": "DOWN",
            "before": h0.tolist(), "after": h1.tolist(), "anim_chain": [],
            "level_completed": False, "game_over": True, "automatic": False, "action_num": 3,
        })
        assert host.journal.library.current_attempt == 3 and host.journal.library.step_in_attempt == 0
        host.close()
    finally:
        shutil.rmtree(tmp_host, ignore_errors=True)

    # (11) sandbox_bridge wrapped action() blocks execution in CONSOLIDATION_READONLY
    tmp_sb = tempfile.mkdtemp(prefix="cpsa_v21_sb_")
    try:
        exports_ro, _ = build_bridge({
            "enabled": True, "journal_dir": tmp_sb, "consolidation_readonly": True,
            "step": lambda p: {"executed": True}, "grid": lambda: h0, "level": lambda: 2,
        })
        try:
            exports_ro["action"]("RIGHT")
            assert False, "Wrapped action() must raise RuntimeError during CONSOLIDATION_READONLY"
        except RuntimeError as e:
            assert "CONSOLIDATION_READONLY" in str(e)
    finally:
        shutil.rmtree(tmp_sb, ignore_errors=True)

    print("ALL 11 CPSA 2.1 REGRESSION GROUPS PASSED!")


def test_cpsa_v2_2_mbel_laf_upgrades():
    """Verify the 3 CPSA 2.2 architectural upgrades derived from MBEL & LAF:
      1. Per-operator confirmation chi_conf(a) (MBEL §3.4) & strict_operator_confirmation in search()
      2. Hardware Non-Promotion / Write-Back guard on save_invariants (MBEL §2.1 & LAF §12 C2)
      3. Zero-cost Reachable Interaction Frontier when search() returns None + explore_frontier() (LAF §5.2)
    """
    import tempfile
    from inference.agent.cpsa.sandbox_bridge import build as build_bridge

    # (1) Per-operator confirmation chi_conf(a) & strict_operator_confirmation
    env, wmlib, ledger, store, events, api = _make_env()
    api["act_expect"](["RIGHT", "DOWN"], expect=None)
    wmlib.record_transition("SPACE", env["board"].copy(), env["board"].copy())
    sim_good = _sim_good()
    v = api["check_sim"](sim_good)
    assert "RIGHT" in v["confirmed_operators"] and "DOWN" in v["confirmed_operators"]
    assert "SPACE" in v["unconfirmed_noop_operators"]

    env2, wmlib2, ledger2, store2, _, api2 = _make_env()
    api2["act_expect"](["RIGHT"], expect=None)  # moves to (10, 11)
    wmlib2.record_transition("SPACE", env2["board"].copy(), env2["board"].copy())  # SPACE is OMIT at (10, 11)
    api2["act_expect"](["DOWN"], expect=None)   # moves to (11, 11)
    def sim_hallucinated_space(frame, act):
        f = sim_good(frame, act)
        if act == "SPACE" and (11, 11) in f.find(9):
            f.grid[11, 11] = 0
            f.grid[10, 13] = 9
        return f
    goal_good = lambda f: (10, 13) in f.find(9)
    plan_loose = api2["search"](
        env2["board"], sim_hallucinated_space, goal_good,
        actions=("UP", "DOWN", "LEFT", "RIGHT", "SPACE"),
        strict_operator_confirmation=False,
    )
    assert plan_loose == ["SPACE"]
    plan_strict = api2["search"](
        env2["board"], sim_hallucinated_space, goal_good,
        actions=("UP", "DOWN", "LEFT", "RIGHT", "SPACE"),
        strict_operator_confirmation=True,
    )
    assert plan_strict == ["RIGHT", "RIGHT", "UP"], f"Expected verified spatial route, got {plan_strict}"

    # (2) Write-Back Non-Promotion Guard on save_invariants (MBEL §2.1 & LAF §12)
    tmp_wb = tempfile.mkdtemp(prefix="cpsa_v22_wb_")
    try:
        b_init = np.zeros((64, 64), dtype=np.int8)
        b_init[10, 10] = 9
        exports_wb, final_wb = build_bridge({
            "enabled": True, "journal_dir": tmp_wb, "consolidation_readonly": False,
            "step": lambda p: {"executed": True, "level_completed": False, "game_over": False, "level": 1},
            "grid": lambda: b_init, "level": lambda: 1,
        })
        exports_wb["save_invariants"](
            mechanics={"teleporter": "[checked] blue block teleports"},
            win_condition="[checked] touch yellow block",
        )
        st_wb = final_wb()["invariant_store_state"]
        assert st_wb["confirmed_mechanics"]["teleporter"].startswith("[assumed]")
        assert st_wb["win_condition"].startswith("[assumed]")
    finally:
        shutil.rmtree(tmp_wb, ignore_errors=True)

    # (3) Zero-cost Reachable Interaction Frontier when search() returns None
    env3, wmlib3, ledger3, store3, _, api3 = _make_env()
    env3["board"][12, 11] = 11
    api3["act_expect"](["RIGHT", "DOWN"], expect=None)
    unreachable_goal = lambda f: (50, 50) in f.find(9)
    res_none = api3["search"](env3["board"], sim_good, unreachable_goal, max_depth=4)
    assert res_none is None
    assert len(wmlib3.last_search_frontier) >= 1
    events_str = " | ".join(item["event"] for item in wmlib3.last_search_frontier)
    assert "color 4" in events_str or "color 5" in events_str or "color 11" in events_str, events_str

    print("ALL 3 CPSA 2.2 MBEL/LAF ARCHITECTURAL UPGRADES PASSED!")


if __name__ == "__main__":
    run_acceptance_suite()
    test_diff_text()
    test_utils()
    test_expect_cells()
    test_journal_roundtrip()
    test_guards_corridor_and_cycle()
    test_guards_clicks_and_escalation()
    test_advisory_tier()
    test_bridge_and_consolidation()
    test_hud_report()
    test_redundant_reset_no_typeerror()
    test_frameview_and_arc_color_chars()
    test_boundary_projection_quotient_mode()
    test_reasoning_both_keys()
    test_pre_level_swap_grid_no_false_before()
    test_vacuous_truth_and_undecided_v4()
    test_cpsa_v2_1_regressions()
    test_cpsa_v2_2_mbel_laf_upgrades()
    print("ALL CPSA 2.2 ACCEPTANCE + REGRESSION TESTS PASSED!")
