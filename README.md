# UAV U2GL Traffic Scheduling

This repository contains a working baseline for the time-varying link resource allocation problem described in `problem_definition.md`. It provides:

- A greedy scheduler that always produces a feasible schedule.
- A deterministic lookahead scheduler inspired by the NTN temporal-graph paper.
- A MILP backend (PuLP/CBC) that reaches near-optimal scores on small instances.
- CLI entry points, example data, and documentation to help you iterate quickly.

## Quick Start

- Python: 3.10+
- Optional: `numpy` (not required for the baseline)

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate random instances:

```bash
# 5x5 grid, 20 flows, T=60, with hotspots and seed
python -m src.generator -M 5 -N 5 -F 20 -T 60 --seed 42 \
  --hotspots 2 --hotspot-boost 2.5 -o examples/gen1.in

# Minimal: 3x3 grid, 2 flows, T=10 (close to the example)
python -m src.generator -M 3 -N 3 -F 2 -T 10 -o examples/gen_small.in
```

Run the schedulers on the example input (file input or stdin):

```bash
# Greedy (default)
python -m src.main -i examples/example1.in -o out.txt

# Temporal (lookahead)
python -m src.main --solver temporal -i examples/example1.in -o out_temporal.txt

#
python -m src.main -i examples/gen1.in gen1_out_temporal.txt


# MILP (PuLP/CBC). Requires 'pulp' installed.
python -m src.main --solver milp -i examples/example1.in -o out_milp.txt

# Or via stdin/stdout
python -m src.main < examples/example1.in > out.txt

# Or run as a script directly
python src/main.py -i examples/example1.in -o out.txt
```

Score the produced output:

```bash
python -m src.scorer examples/example1.in out.txt
```

## Repository Structure

- `src/main.py` - CLI entry reading stdin and writing stdout.
- `src/io.py` - Input parsing and output formatting.
- `src/model.py` - Core dataclasses for UAVs, flows, and the problem.
- `src/bandwidth.py` - U2GL bandwidth model with phase.
- `src/scheduler.py` - Baseline greedy scheduler.
- `src/scheduler_temporal.py` - Time-expanded lookahead scheduler inspired by NTN deterministic routing.
- `src/solvers/milp_pulp.py` - MILP solver backend using PuLP (CBC).
- `src/scorer.py` - Scoring implementation per the problem statement.
- `examples/` - Example input and reference output.
- `docs/brainstorm.md` - Baseline design notes and improvement ideas.
- `docs/ntn_integration_notes.md` - Mapping and integration notes for the NTN paper.

## Baseline Greedy Heuristic

For each second:

- Compute U2GL capacities for all UAVs using the periodic pattern.
- Consider flows that have started and still have remaining traffic.
- For each active flow, inspect a precomputed top-`K` landing set ordered by average capacity and distance.
- Pick the best landing using current capacity, Manhattan distance penalty, and a small stickiness bonus to avoid churn.
- Allocate up to the minimum of (flow remaining, landing capacity) and update state.

This ensures feasibility while remaining fast enough for the large-instance limits (70×70 grid, 500 seconds, 5000 flows).

## Temporal Scheduler (New)

- Builds a time-expanded graph over a configurable horizon (default 12 seconds) and scores candidate `(time, landing)` pairs jointly.
- Uses a global priority queue so the highest-value slots (early, short distance, reusable landings) are filled first.
- Commits only the current second, then replans with updated state—mirroring the deterministic routing cycle-by-cycle refinement highlighted in the NTN paper.
- Landing reuse earns an explicit bonus, while opening a brand-new landing incurs a penalty tied to the 1/k objective component.

Key options:

- `--solver temporal` activates the new scheduler.
- `--horizon` controls the lookahead window in seconds (default 12).
- `--top-k` limits the landing candidates per flow (shared with the greedy solver; lower values keep planning fast).
- `--stickiness`, `--temporal-reuse`, and `--temporal-penalty` tune the reuse/penalisation trade-offs.

## MILP Solver

`src/solvers/milp_pulp.py` formulates the full-horizon problem as a MILP with per-slot landing binaries and a landing-count linearisation. It is useful for benchmarking or small instances (CBC, SCIP, or commercial solvers). Enable it via `--solver milp`.

## Assumptions and Notes

- Distance is Manhattan (`L1`) hops on the M×N grid.
- The aerial mesh routing is abstracted away; only U2GL capacity is modelled explicitly.
- The bandwidth pattern is `[0, 0, 0.5, 1, 1, 1, 1, 1, 0.5, 0]`, shifted by each UAV phase (`I+`).
- Output includes only time slots with positive transmission rates for each flow.

## Ideas for Further Work

- Hybridise the temporal scheduler with small per-slot linear programs to share landing capacity more fairly when contention is high.
- Incorporate explicit landing-change counters (budgeted or penalised) instead of heuristic bonuses.
- Investigate batching flows with similar rectangles to reduce repeated candidate scoring.
- Explore solver warm-starts: feed the temporal schedule into the MILP backend to accelerate optimality proofs.

