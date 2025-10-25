## Huawei Traffic Allocation Solver (Python)

This repository hosts a Python-first implementation of the UAV-to-ground traffic allocation challenge described in `problem_definition.md`. The solver models per-second U2GL capacities, enforces Manhattan travel before landing, and optimises flow allocations with PuLP/CBC. All functionality is exposed through a single CLI that accepts either standard input (for the evaluator) or explicit file paths.

### Requirements
- Python 3.10 or newer
- Dependencies listed in `requirements.txt` (`pulp` is optional but strongly recommended for the MILP path)

Install the runtime:
```bash
pip install -r requirements.txt
```

### Running the solver
The canonical entry point is `python -m src.cli`. When no `--input` flag is given the solver reads the problem definition from `stdin`, matching the organiser's evaluation harness. Plain-text `.in` files and JSON payloads are both supported.

```bash
# stdin → stdout (evaluation mode)
python -m src.cli < examples/example1.in

# explicit file input
python -m src.cli --input examples/example1.in

# write canonical .out file and validate legality
python -m src.cli --input examples/example1.in --output-text example1.out --validate

# emit JSON alongside the text schedule and raise the global time limit
python -m src.cli --input examples/example1.in \
  --output-text example1.out \
  --output-json example1.json \
  --time-limit 10
```

All commands print the final score to `stderr`. If `pulp` is unavailable the solver falls back to the greedy warm start while preserving rule compliance.

### Validation
Schedules can be checked against the official rules (single landing per time slot, zone membership, earliest-arrival, and capacity constraints) via:
```bash
python -m src.cli --input examples/example1.in --validate
```
Alternatively, import `validate_schedule` from `src.validation` inside your own notebooks or scripts.

### Scoring existing schedules
Use `python -m src.score_cli` to evaluate previously generated `.out` files:
```bash
python -m src.score_cli --input examples/example1.in --schedule examples/example1.out --validate

python -m src.score_cli --input examples/gen1.in --schedule examples/julia_gen1.out --validate

python -m src.score_cli --input examples/gen1.in --schedule examples/gen1V3.out --validate

python -m src.score_cli --input examples/gen1.in --schedule examples/gen1p.out --validate


```
The command prints the total score to stdout and, with `--output-json`, produces a JSON summary (assignments + per-flow metrics). Programmatic access is available through `src.scorer.score_from_files`.

### Project layout
```
src/
  __init__.py         # exposes solve_problem, score helpers, validation
  capacity.py         # 10-slot periodic capacity model
  candidates.py       # earliest-arrival windows and time-slot pruning
  cli.py              # solver command-line entry point (stdin aware)
  clustering.py       # landing-zone overlap clustering
  heuristics.py       # greedy warm start (fallback when MILP unavailable)
  models.py           # dataclasses for grids, UAVs, flows, schedules
  parser.py           # .in / JSON readers
  postprocess.py      # scoring, text/JSON emitters, and .out parsing
  score_cli.py        # CLI to score and validate existing schedules
  scorer.py           # reusable scoring functions
  solver.py           # PuLP/CBC MILP per cluster
  validation.py       # organiser legality checks
tests/
  conftest.py         # adds repository root to sys.path
  test_solver.py      # unit tests covering parsing, capacity, solver, scoring
examples/
  example1.in         # sample instance from the documentation
  example1.out        # reference schedule for scoring tests
```

### Development
- Run the unit suite with `pytest`.
- Extend `tests/test_solver.py` when adding new modules or rules.
- Use `conversion_to_python.md` for the detailed migration roadmap pulled from the Julia prototype.

### Troubleshooting
- **`ModuleNotFoundError: pulp`** – install the solver dependency (`pip install pulp`). Without it the CLI still returns a schedule but scores will reflect the greedy warm start.
- **Validation failures** – re-run with `--validate` (solver) or `python -m src.score_cli --validate` to obtain the exact rule that was violated (e.g., multiple landings in the same slot, out-of-zone coordinates).

This README is intentionally concise for quick onboarding. Refer to `problem_definition.md`, `Q_and_A.txt`, and the inline module docstrings for full algorithmic detail.
