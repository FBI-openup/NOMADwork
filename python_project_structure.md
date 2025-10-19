# Python Project Structure (snapshot)

This summarizes the current Python repository layout. Refresh it whenever modules move or new tooling is introduced.

## Repository map
```
.
├── README.md                   - Challenge overview, usage instructions, and quick start.
├── requirements.txt            - Python dependencies (solver backends, CLI helpers).
├── problem_definition.md       - Formal problem statement, inputs, outputs, constraints.
├── NTN paper.md                - Research excerpts guiding the solver architecture.
├── next_steps.md               - Working backlog of experiments and follow-up tasks.
├── gen1_out_temporal.out       - Sample solution from the temporal scheduler run.
├── out.txt                     - Latest solver output kept for quick diffing.
├── python_project_structure.md - Repository map (this document).
├── docs/                       - Design notes and planning artifacts.
│   ├── brainstorm.md           - Early ideation on solver strategies and heuristics.
│   ├── ntn_integration_notes.md- Integration considerations for the NTN stack.
│   └── solver_plan.md          - Milestones and architecture roadmap for solver evolution.
├── examples/                   - Input/output fixtures for manual runs and regression checks.
│   ├── README.md               - Replay instructions and context for each example dataset.
│   ├── example1.in             - Baseline challenge instance.
│   ├── example1.out            - Expected solution for the baseline instance.
│   ├── example1_cli.out        - CLI transcript demonstrating solver invocation.
│   ├── gen1.in                 - Generated instance used for MILP vs heuristic comparison.
│   ├── gen1_greedy.out         - Greedy solver output for `gen1.in`.
│   ├── gen1_milp.out           - MILP solver output for `gen1.in`.
│   ├── out_greedy.txt          - Additional greedy solver artifact.
│   └── out_milp.txt            - Additional MILP solver artifact.
├── src/                        - Python package housing the solver implementation.
│   ├── __init__.py             - Package marker; re-exports convenience helpers when needed.
│   ├── main.py                 - CLI entry point orchestrating parsing, solver selection, reporting.
│   ├── io.py                   - File I/O helpers for reading challenge instances and writing solutions.
│   ├── generator.py            - Scenario generator for synthetic workloads and stress tests.
│   ├── model.py                - Domain entities (nodes, links, requests, capacities, horizon).
│   ├── bandwidth.py            - Link capacity calculations and feasibility checks.
│   ├── scheduler.py            - Core static scheduler coordinating solver passes.
│   ├── scheduler_temporal.py   - Time-expanded scheduler variant for dynamic constraints.
│   ├── scorer.py               - Objective scoring and constraint validation utilities.
│   └── solvers/                - Pluggable solver implementations behind a shared interface.
│       ├── __init__.py         - Solver registry and factory helpers.
│       ├── interface.py        - Abstract solver contract plus adapter utilities.
│       ├── milp_pulp.py        - MILP formulation implemented with PuLP (primary solver).
│       └── milp_stub.py        - Lightweight heuristic or placeholder solver.
├── tests/                      - Automated regression tests.
│   └── test_example.py         - Smoke test covering the full pipeline on `example1.in`.
├── Datacom - Time-varying link resource allocation algorithm.pdf
├── enhancing NTN TEG.pdf
└── huawei agorize 2025 france.tar.zst
```

## Module responsibilities
- `src/main.py`: Parses CLI args, loads instances via `io.py`, routes to the chosen solver, and emits outputs.
- `src/io.py`: Reads `.in` challenge files, writes `.out` submissions, and handles extra JSON/CSV exports.
- `src/generator.py`: Builds synthetic traffic scenarios to probe solver performance.
- `src/model.py`: Defines the typed objects for requests, nodes, links, and scheduling horizons.
- `src/bandwidth.py`: Centralizes bandwidth math and link-capacity feasibility logic shared across solvers.
- `src/scheduler.py`: Static scheduling coordinator that sequences solver calls and scorer validations.
- `src/scheduler_temporal.py`: Temporal extension relying on time-expanded graphs for dynamic constraints.
- `src/scorer.py`: Objective calculation and constraint enforcement; reused in tests and CLI reporting.
- `src/solvers/interface.py`: Declares the solver API (`solve(instance) -> schedule`) and registry helpers.
- `src/solvers/milp_pulp.py`: PuLP-backed MILP solver; primary optimization engine.
- `src/solvers/milp_stub.py`: Heuristic/fallback solver for fast experimentation or baseline comparisons.
- `tests/test_example.py`: Regression harness ensuring the solver produces the expected output for `example1.in`.

## Working notes
- Use the artifacts in `examples/` to validate behaviours after solver tweaks—compare outputs with `diff` for quick regressions.
- PDF references at the repository root provide background context; they do not influence the runtime code path.
- When adding a solver, implement it under `src/solvers/`, register it in `__init__.py`, and extend tests to cover the new mode.
- Keep generated `.out` files either inside `examples/` or in a dedicated `artifacts/` directory so they are easy to clean between runs.
