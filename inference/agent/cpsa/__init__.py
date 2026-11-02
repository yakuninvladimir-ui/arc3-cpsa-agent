"""CPSA 2.0 core package.

Import convention (STRICT): the parent directory of this ``cpsa`` package is
put on ``sys.path`` (by the harness orchestrator and by the sandbox bootstrap),
and modules are imported ONLY as ``cpsa.<module>`` / ``from cpsa import ...``.
Never import them as ``inference.agent.cpsa.<module>`` in the sandbox — that
would create a second module identity and break isinstance checks.

All modules are self-contained: standard library + numpy only, no project
imports, no ``from __future__`` statements.
"""
