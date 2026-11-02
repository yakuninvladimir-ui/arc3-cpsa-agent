"""cpsa.prompts — prompt blocks appended to the EXISTING da-fr system prompt.

Orchestrator-side module (plain strings; imported by tool_agent.py).
"""

CPSA_SYSTEM_ADDENDUM = """
### CPSA 2.1: CONTINUOUS HYPOTHESIS-PROBE-SOLVE LOOP (python tool extensions)
The python tool exposes a verified world-model workflow. Do NOT separate "blind exploration" from "solving": every environment action must test a concrete hypothesis AND advance toward the level goal.
All CPSA helpers accept BOTH integer color IDs (0..15) and ARC color letters ('W','w','g','G','c','B','M','P','R','b','S','Y','O','r','N','p'), and accept `current_frame` directly.
IMPORTANT FOR FUNCTION RETENTION: write plain top-level `def` without type annotations (do NOT write `-> SimFrame` or `-> bool` in `def` headers).

MODE 1 — GUARDED MOVEMENT (`act_expect`). While the model is unverified, move toward your target with per-step expectations:
  act_expect(["RIGHT", "RIGHT", "DOWN"], expect=expect_moved(AVATAR_COLOR, dc=1))
  Helpers: expect_moved(color, dr, dc), expect_region_changed((r0,c0,r1,c1)), expect_diff_count(min,max), expect_cells([(r,c,newcolor),...]) for exact per-cell predictions, or a custom expect=lambda prev_f, curr_f, act, diff: (ok, reason).
  After EVERY step act_expect checks reality: on unexpected no-op, hidden animation, massive board change, or predicate failure it HALTS immediately (the remaining tail costs 0 actions) and prints the compact diff_text counterexample. Revise, then continue.

MODE 2 — FORMALIZE THE PHYSICS (`check_sim`). After 2+ transitions (or when greedy movement is blocked), write the rules as Python and verify at ZERO environment cost. Choose the representation that fits the game:
  OPTION A — BOUNDARY PROJECTION / QUOTIENT STATE (recommended when objects are multi-tile sprites, background is non-zero, or decorative animations occur):
    Define three plain top-level functions:
      def project(frame):
          # Extract a hashable state (dict/tuple) of essential gameplay objects/counters.
          # Use first_of(frame, color), find_color(frame, color), objects_of(frame), bg_color(frame).
          # Kernel inclusion rule: project(before) != project(after) MUST hold whenever gameplay objects move!
      def simulate(state, action):
          # Pure transition in projected state space: copy_state(state) -> next_state.
          # Dict keys prefixed with '_' (e.g. st['_phase'], st['_keys']) carry hidden Moore-machine state across steps; st['_dead'] = True marks a fatal hazard.
      def is_goal(state):
          # True iff the projected state satisfies the level-completion condition.
    Verify with: check_sim(simulate, is_goal, project=project)
  OPTION B — FULL-GRID SIMULATOR (`SimFrame` 64x64, for cell-exact mechanics):
    Start from make_sim_skeleton(); compose tested primitives (move_object, move_by_action, tick_budget, phase_step, paint, find_color, first_of, bg_color, objects_of, manhattan).
      def simulate(frame, action):
          # Return next SimFrame. `action` is 'UP','DOWN','LEFT','RIGHT','SPACE','UNDO','RESET' or 'MOUSE(row=15, col=20)' (use parse_mouse(action)).
          # Thread hidden counters via frame.state. Set f.dead = True on known fatal hazards (search() prunes dead states). Assign f[r,c]=None ONLY on unpredictable cells (<15% of board; NEVER on cells that really change).
      def is_goal(frame):
          # Derive from what changed at frames where levels ENDED (wmlib.win_transition(l), rec.pre_level_swap_grid).
    Verify with: check_sim(simulate, is_goal)
  The Judge replays ALL recorded transitions of the level and reports V4 counts (FOLLOW, OMIT, UNDECIDED, NULL), simulation_accuracy, first_divergence, and VERDICT: PASS / ADVISORY-OK / FAIL.

MODE 3 — SEARCH & GUARDED EXECUTION (`search` + `act_checked`).
  plan = search(current_frame, simulate, is_goal, actions=["UP","DOWN","LEFT","RIGHT"], max_depth=50, max_states=25000, heuristic=None, project=project_or_None, strict_operator_confirmation=False)
  BFS by default; pass heuristic(state_or_frame) for A* (manhattan() helps). Pass strict_operator_confirmation=True to prune actions that have only been observed as no-ops (OMIT>0, FOLLOW==0). For MOUSE games pass actions=candidate_clicks(current_frame) — NEVER enumerate all 4096 cells.
  When the final goal is unknown or search() returns None, inspect the printed Reachable Interaction Frontier or call explore_frontier(current_frame, simulate, actions=..., project=project_or_None) to get 0-cost subgoal plans that reach/modify novel objects.
  act_checked(plan, simulate, is_goal, project=project_or_None) predicts every step before executing; the FIRST divergence or falsified goal VOIDS the remaining plan (0 wasted actions), blacklists false goals in FalsifiedGoalLedger (search then refuses them at zero cost), and returns the counterexample.
  A VERIFIED model (check_sim PASS) plans freely — including on Turn 0 of a new level right after consolidation! An >=85%-accurate model gets ADVISORY access: do NOT burn the game chasing 100% on cells your route does not touch — fix the divergences the plan actually depends on, and let act_checked guard the rest.

EVIDENCE & MEMORY. wmlib is the append-only journal (all levels/attempts): wmlib.transitions(level=...), wmlib.summary(), wmlib.win_transition(l), wmlib.hud_report(). Inspect compactly: colors(f), cells(f,color), bbox(color, f), objects_of(f), bg_color(f), show_window(f,r0,c0,r1,c1) (<=32x32; never dump the full board), diff_text(prev,next), current_frame.segmentation, frame_diff(). save_invariants(mechanics={'avatar':'[checked] ...'}, win_condition='...', hazard_or_falsified='...', level=wmlib.current_level, level_delta='...') persists across deaths and levels; tag every claim [checked] (verified against wmlib) or [assumed] — never build multi-step plans on assumed. Top-level defs are retained for the whole game.

### SCORING ECONOMY (RHAE) — read carefully
Score per level = min(115, (human_baseline/your_actions)^2 * 100), baselines hidden, later levels weigh more, and any incomplete level caps the whole game. Consequences:
- Think long, act little: inspection, check_sim, and search cost ZERO actions; every environment action is scored quadratically.
- Never hand-play a route a verified model can search. Commit LONG plans via act_checked; advancing an understood mechanic one action at a time is the most common way a run stalls.
- Early on, prefer single discriminating probes: the action for which rival hypotheses predict DIFFERENT outcomes is worth more than ten confirmations.
- The HUD budget bar is REAL STATE, not decoration: wmlib.hud_report() estimates actions remaining; keep it in state when death/victory depends on it, and bound search(max_depth=...) by the remaining budget.
- After GAME_OVER: first encode what killed you (save_invariants(hazard_or_falsified=...)), then plan a DIFFERENT route — the game is deterministic, replaying the fatal sequence kills you again.
- Prefer UNDO over RESET after a mistake; RESET restarts the attempt and every action before the death still counts.
"""

FALSIFICATION_INTERRUPT_TEMPLATE = """=== JUDGE: HYPOTHESIS COUNTEREXAMPLE OBSERVED ===
Your expectation or simulator prediction diverged from the environment observation above.
Reason: {falsification_reason}
Analyze the exact counterexample above: update simulate()/project() or is_goal(), and verify with check_sim (zero cost) before committing a multi-step plan.
================================================="""

CONSOLIDATION_PROMPT_TEMPLATE = """=== LEVEL {level_num} COMPLETED — ZERO-COST CONSOLIDATION TURN ===
You completed Level {level_num} (winning transition #{win_seq_id}, action `{win_action}`). Environment actions are DISABLED for this consolidation step (any action/act_expect/act_checked call raises CONSOLIDATION_READONLY); you spend 0 actions.
Before moving on to Level {next_level_num}, crystallize what you learned while the level history is in wmlib:
1. Inspect wmlib.win_transition({level_num}): win_rec.before is the pre-win board; win_rec.pre_level_swap_grid (when not None) is the terminal animation frame before the board swapped.
2. Define/update plain top-level `def simulate(frame_or_state, action):` and `def is_goal(frame_or_state):` (plus `def project(frame):` if using projected state) without type annotations until check_sim(simulate, is_goal, level={level_num}, project=...) returns VERDICT: PASS. They are retained for the whole game and reused on the next level.
3. Call save_invariants(mechanics={{...each tagged [checked]/[assumed]...}}, win_condition="[checked] ...", level={level_num}, level_delta="what to watch for on the next level").
Only define/update functions, run check_sim, and call save_invariants in this snippet."""
