# UAV U2GL Traffic Scheduling (Baseline)

This repository contains a baseline solution for the time‑varying link resource allocation problem described in `problem_definition.md`. It provides:

- A simple, working greedy scheduler to generate a feasible schedule.
- A scoring tool to evaluate a produced schedule for a given input.
- CLI entry point, example data, and documentation to help you iterate on algorithms.

The intent is to give you a solid starting point so you can focus on improving the algorithmic strategy (e.g., smarter assignment, dynamic programming, or learning‑based approaches) without spending time on boilerplate and plumbing.

## Quick Start

- Python: 3.10+
- Optional: `numpy` (not required for baseline)

Install dependencies:

```
pip install -r requirements.txt
```

Run the scheduler on the example input (file input or stdin):

```
# Greedy (default)
python -m src.main -i examples/example1.in -o out.txt

# Or via stdin/stdout
python -m src.main < examples/example1.in > out.txt

# Or run as a script directly
python src/main.py -i examples/example1.in -o out.txt

# MILP (PuLP/CBC). Requires 'pulp' installed.
python -m src.main --solver milp -i examples/example1.in -o out_milp.txt
```

Score the produced output:

```
python -m src.scorer examples/example1.in out.txt
```

## Repository Structure

- `src/main.py` — CLI entry reading stdin and writing stdout.
- `src/io.py` — Input parsing and output formatting.
- `src/model.py` — Core dataclasses for UAVs, flows, and the problem.
- `src/bandwidth.py` — U2GL bandwidth model with phase.
- `src/scheduler.py` - Baseline greedy scheduler.
- `src/solvers/milp_pulp.py` - MILP solver backend using PuLP (CBC).
- `src/scorer.py` — Scoring implementation per the problem statement.
- `examples/` — Example input and reference output.
- `docs/brainstorm.md` — Design notes, assumptions, and improvement ideas.

## Baseline Heuristic (Summary)

At each time slot t:

- Compute U2GL capacities for all UAVs.
- Consider all flows that have started and are not yet complete.
- For each active flow, greedily select a landing UAV among a precomputed top‑K candidate set using a dynamic utility that balances: current capacity, Manhattan distance penalty, and a small “stickiness” bonus to avoid changing landing points frequently.
- Allocate up to the minimum of (flow remaining, UAV capacity at t) for that second.

This yields a feasible schedule with one landing UAV per flow per time slot, respecting U2GL capacities.

## Assumptions and Notes

- Distance is Manhattan (`L1`) hops on the MxN grid.
- Exact routing within the aerial mesh is abstracted by the distance penalty; the only capacity constraint modeled is at U2GL.
- The bandwidth periodic pattern is: [0, 0, B/2, B, B, B, B, B, B/2, 0] applied with the UAV‑specific phase.
- Output includes only time slots with positive transmission rates for each flow.

## Next Steps (Ideas)

- Better capacity sharing at each UAV (small LP per time step or auction‑based assignment).
- Explicitly model landing point change count to maximize the “Landing UAV Point” score component.
- Predictive lookahead over several slots to align flow bursts with future capacity peaks.
- Tie‑breaking by per‑UAV contention level to avoid oversubscribed hotspots.
- Parallelize per‑time computations and precompute more structure for large instances.
