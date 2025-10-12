# Solver Conversion Plan (MILP/SOS1)

This problem is not convex in its native form due to the per-time-slot single-landing choice and the 1/k landing-point term. A practical exact approach is a MILP with SOS1 constraints and a small number of binaries, solved via branch-and-bound/cuts. Below is a concrete plan.

## Core Modeling

- Indices:
  - Flows `f ∈ F`, time `t ∈ {0..T-1}`, candidate landings `j ∈ J_f` (top-K per flow).
- Data:
  - `cap_{j,t}` from the bandwidth pattern with per-UAV phase.
  - `Q_f`, `t_start,f`, `w_delay(t)=T_max/(t+T_max)`, `w_dist(f,j)=2^{-α·d(f→j)}`.
- Variables:
  - `z_{f,t,j} ≥ 0` — traffic assigned (Mbits) for flow f at time t to landing j.
  - SOS1 per (f,t): at most one `z_{f,t,j}` positive (or binaries `y_{f,t,j} ∈ {0,1}` with `∑_j y_{f,t,j} ≤ 1` and `z_{f,t,j} ≤ Q_f y_{f,t,j}`).
  - `u_{f,j} ∈ {0,1}` — landing j used at any time; link via `z_{f,t,j} ≤ Q_f u_{f,j}`.
  - Optional `v_{f,k} ∈ {0,1}` — encode landing changes count k to linearize the 1/k term.
  - `s_f ∈ [0,1]` — delivered fraction; `s_f ≤ (∑_{t,j} z_{f,t,j})/Q_f`.
- Constraints:
  - Capacity: `∑_f z_{f,t,j} ≤ cap_{j,t}` for all j,t.
  - Flow budget: `∑_{t≥t_start,f, j} z_{f,t,j} ≤ Q_f`.
  - Time feasibility: disallow t < `t_start,f`; skip (t,j) where `cap_{j,t}=0`.
  - Landing usage: `z_{f,t,j} ≤ Q_f u_{f,j}`.
  - Optional landing-count linearization: `∑_j u_{f,j} = ∑_k k·v_{f,k}`, `∑_k v_{f,k}=1`.
- Objective (linear): maximize weighted sum per flow
  - `0.4·s_f + 0.2·∑_{t,j} w_delay(t)·z_{f,t,j}/Q_f + 0.3·∑_{t,j} w_dist(f,j)·z_{f,t,j}/Q_f + 0.1·∑_k (1/k)·v_{f,k}`
  - Then weight flows by `Q_f/ΣQ` globally (constant scaling).

## Scale Management

- Top-K candidates per flow (K=10–30) by static score `B·2^{-α·d}`.
- Drop all (t,j) with zero capacity due to phase (only 6/10 slots active).
- Time windowing: optionally cap final times where `w_delay(t)` is negligible.
- Dominance pruning: remove candidate j if dominated by j′ (higher B and no longer distance).
- Warm start: seed MILP with greedy schedule from `src/scheduler.py`.

## Solver Choice

- Best performance: Gurobi / CPLEX (commercial).
- Open source: SCIP handles SOS1 and branching well; CBC is possible but weaker with SOS1.
- Python modeling: Pyomo or OR-Tools for portability; PuLP works but SOS1 support is limited.

## Implementation Plan

1. Add `src/solvers/interface.py` defining a `solve(problem) -> schedules` interface.
2. Add `src/solvers/milp_stub.py` that constructs the reduced instance (top-K, active times) and returns NotImplemented until a solver backend is plugged.
3. Add CLI flags in `src/main.py` to choose `--solver greedy|milp` (default greedy). If `milp`, call stub and print a helpful error about installing a solver.
4. Provide documentation on enabling Gurobi/SCIP backends and how to warm start using the greedy schedule.

## Warm Start Strategy

- For each (f,t), set the chosen landing’s `z_{f,t,j}` to the greedy amount; mark that SOS1 member as preferred.
- For solvers with MIP starts, set corresponding `y_{f,t,j}` and `u_{f,j}`.
- This typically reduces B&B tree size substantially.

