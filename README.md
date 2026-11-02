# ARC-AGI-3: Continuous Python-Symbolic Agent (CPSA 2.3)

State-of-the-art autonomous symbolic game-solving agent developed for the **ARC Prize 2026 (ARC-AGI-3)** competition.

Built upon and extending foundational open-source solutions from:
* **Tufa Labs** — [duck-harness](https://github.com/Tufalabs/duck-harness) (TAAF evaluation stack & tool harness architecture)
* **Daniel Franzen** — [arc-agi-3-solution](https://github.com/da-fr/arc-agi-3-solution) (groundbreaking ARC-AGI-3 solving methodology and agent concepts)

CPSA 2.3 unifies large-scale autoregressive reasoning with dynamic symbolic program synthesis, verifiable Moore World Models, $V_4$ semantic judges, priority stream rotation, and cross-level persistent invariant retention.

---

## Key Architectural Components

### 1. Three-Mode Execution Engine
* **Mode 1 (Immediate Guarded Exploration `act_expect`):** Evaluates local state expectations before each action and halts instantly on expectation divergence without wasting planned move budgets.
* **Mode 2 (Moore World Model & $V_4$ Judge `check_sim`):** Reconstructs verifiable state transition physics (`simulate`), boundary projections ($\pi_K$ quotient abstraction `project`), and terminal goal conditions (`is_goal`), dynamically verified against past gameplay history in `wmlib` under four-valued semantics (`FOLLOW`, `OMIT`, `UNDECIDED`, `NULL`).
* **Mode 3 (Graph Search & Guarded Execution `search` + `act_checked`):** Performs deterministic graph search on verified physics with hazard pruning, preventing goal hallucinations via runtime latching in `FalsifiedGoalLedger`.

### 2. Knowledge Retention & Cross-Level Transfer
* **`GameInvariantStore`:** Persists verified mechanics (`[checked]` claims, avatar identities, hazard boundaries) across attempts and levels in a structured, append-only memory block.
* **Zero-Cost Level Consolidation (`ARC3_CONSOLIDATION_TURN=1`):** End-of-level reflection turns (priority 0) synthesize and generalize cross-level invariants without consuming gameplay actions.
* **Guaranteed Persistent Store Injection:** Custom context drainage preserves persistent invariant memory across any token pruning or level handover events.

### 3. High-Throughput Priority Scheduling
* **`_PriorityGate` (16 concurrent streams):** Dynamically arbitrates execution across games, allocating GPU prefill and generation bandwidth to games with the highest progress probability.
* **Monotonic Progress Bonus:** Gives logarithmic priority acceleration to games reaching deeper levels.
* **Live Token Accounting & Handover:** Yields analyzer turns gracefully when token budgets or level transitions occur, preventing thread starvation.

---

## Directory Structure

```text
arc3-cpsa-agent/
├── pyproject.toml              # Modern PEP 517/621 packaging
├── requirements.txt            # Python dependencies
├── .env.example                # CPSA 2.3 limits, hyperparameters, and flags
├── run.py                      # Root CLI runner
├── inference/                  # Core CPSA 2.3 Inference Agent
│   ├── agent/
│   │   ├── cpsa/               # World model, wmlib, invariants, sandbox bridge
│   │   ├── tool_agent.py       # Main ToolAgent analyzer
│   │   ├── priority_scheduler.py# Priority gate & 16-stream pacing
│   │   └── ...
│   ├── framework/
│   │   ├── solver.py           # HarnessSolver (implements TAAF Solver)
│   │   └── run.py              # Benchmark execution orchestrator
│   └── utils/                  # Retained functions, frame diffs, vision context
├── taaf/                       # TAAF Competition Arcade & Benchmark Framework
├── viewer/                     # Interactive game & transcript visualizer
└── tests/
    └── test_cpsa_acceptance.py # CPSA acceptance & regression test suite
```

---

## Quickstart

### 1. Installation
Clone this repository and install dependencies in a Python 3.10+ virtual environment:

```bash
git clone https://github.com/<your-username>/arc3-cpsa-agent.git
cd arc3-cpsa-agent

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### 2. Configuration
Copy `.env.example` to `.env` and set your model endpoint and settings:

```bash
cp .env.example .env
```

Key environment variables:
* `LOCAL_ANALYZER_BASE_URL`: URL of your inference server (e.g. SGLang or vLLM at `http://127.0.0.1:8001/v1`).
* `LOCAL_ANALYZER_MODEL`: Model identifier (e.g. `flashnext` or `Qwen/Qwen2.5-Coder-32B-Instruct`).
* `LOCAL_ANALYZER_CONTEXT_WINDOW`: Target context window size (e.g. `131072`).
* `ARC3_MAX_ACTIVE_STREAMS`: Concurrency limit for parallel game execution (default: `16`).

### 3. Running the Agent
Run the agent using the root runner:

```bash
python run.py --config configs/inference.json
```

Or run the viewer to inspect past games and transcripts:

```bash
python -m viewer.server --port 8080
```

### 4. Running Tests
Run the test suite:

```bash
pytest
```

---

## Acknowledgements & Attribution

This project builds upon the pioneering work of the ARC-AGI research community. In particular, we gratefully acknowledge:
* **[Tufa Labs](https://github.com/Tufalabs)** for [duck-harness](https://github.com/Tufalabs/duck-harness) and the TAAF benchmark framework.
* **[Daniel Franzen (da-fr)](https://github.com/da-fr)** for [arc-agi-3-solution](https://github.com/da-fr/arc-agi-3-solution), whose foundational ideas, architecture, and exploration strategies inspired this implementation.

---

## License
MIT License.
