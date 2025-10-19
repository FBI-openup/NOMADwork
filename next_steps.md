# Next Steps: Julia Migration And MIP Plan

## Why we need an update
- The original challenge requires scheduling **volumetric flows over multiple seconds**, not assigning an entire flow at a single time. The prior plan treated each flow as a single-shot assignment, which cannot satisfy flows whose volume exceeds per-slot bandwidth.
- Each flow has a declared `t_start`; earliest transmission must respect both `t_start` and the minimum Manhattan travel time to the chosen landing cell.
- U2GL capacity is dictated by the 10-second periodic function `b(I+ + t)` with peak `B` and phase offset `I+` per UAV. We should derive per-(cell, time) capacities directly from that rule instead of leaving them abstract.
- The scoring function combines throughput, delay, distance, and landing-cell churn. The strategy should acknowledge how the optimization approximates or replicates that scoring, rather than using a generic linear cost.

## Problem model (aligned with problem_definition.md)
**Entities**
- `Flows (F)`: attributes `{id, start=(x_s, y_s), t_start, volume Q_total, zone rectangle [m1, n1] to [m2, n2]}`.
- `Landing cells (Z)`: UAV grid cells `(x, y)`; each has a 10-step bandwidth profile defined by `{B, I+}`.
- `Time steps (T)`: discrete seconds `0 ... T-1` from the instance.

**Derived data**
- `travel_time(f, z) = |x_s - x_z| + |y_s - y_z|`.
- `earliest_arrival(f, z) = max(t_start_f + travel_time(f, z), t_start_f)`; transmission before this is forbidden.
- `capacity(z, t) = pattern(B_z, I+_z, t)` where the 10-second pattern is `{0, 0, B/2, B, B, B, B, B/2, 0, 0}` and repeats.

**Decision variables**
- `x[f, z, t] >= 0`: Mbps scheduled for flow `f` through landing cell `z` during second `t`.
- `u[f, z, t] in {0,1}`: 1 if flow `f` uses landing cell `z` at time `t` (prevents simultaneous multi-cell transmission).
- Optional `y[f, z] in {0,1}`: 1 if flow `f` ever uses `z`; useful for landing-point scoring and reporting.

**Constraints**
- Zone membership: set `u[f, z, t] = 0` and `y[f, z] = 0` whenever `z` lies outside the flow's rectangle.
- Time feasibility: `u[f, z, t] = 0` (and therefore `x[f, z, t] = 0`) for any `t < earliest_arrival(f, z)` or `t < t_start_f`.
- Flow completion: `sum_{z,t} x[f, z, t] = Q_total_f`.
- Single landing cell per time step: `sum_{z} u[f, z, t] <= 1` for every `t`. Sequential moves between landing cells become legal because earliest-arrival checks already encode the travel time.
- Link usage to transmission: `x[f, z, t] <= capacity(z, t) * u[f, z, t]`.
- Accumulate visited cells: enforce `u[f, z, t] <= y[f, z]` and, if desired, `y[f, z] <= sum_{t} u[f, z, t]`.
- Ground capacity: `sum_f x[f, z, t] <= capacity(z, t)` for all `(z, t)`.

**Objective (maximize challenge score)**
- Implement the weighted sum directly:
  - Throughput: enforce completion or apply a large penalty if dropping is allowed.
  - Delay: apply `T_max / (t + T_max)` to each scheduled chunk `x[f, z, t]`.
  - Distance: weight by `2^{-alpha * d(f, z)}` using the active landing cell at each `t`, where `alpha = 0.1`.
  - Landing churn: compute `k_f = |{z : y[f, z] = 1}|` (or derive from `u`) and include the `1 / k_f` factor.
- For a linear MIP, build a surrogate objective with calibrated weights on `x`, `u`, and `y`, then validate by scoring solutions via the official formula.

## Hotspot clustering (updated to match volumetric scheduling)
1. Build a bipartite graph between flows and landing cells they are permitted to use.
2. Two flows share a cluster if their zones intersect or are adjacent (optional 8-neighbour dilation for safety).
3. Solve each cluster independently: the constraints only couple flows that share at least one landing cell, so clusters form independent subproblems.
4. Within a cluster, time slots remain coupled by capacity; solving by cluster still reduces MIP size markedly.

## Candidate pruning tuned for volumetric flows
- **Time window truncation:** for each `(f, z)`, generate candidate times `t` within `[earliest_arrival(f,z), earliest_arrival(f,z) + delta_window]`, where `delta_window` starts around 10-20 seconds. Expand only if the flow cannot be completed.
- **Bandwidth-aware pruning:** skip times where `capacity(z, t) = 0`.
- **Dominance:** if two candidate times have identical capacity and a strictly worse delay score, keep only the better one.
- **Aggregation option:** group consecutive seconds with identical capacity into "time blocks" to reduce variables, while tracking how much of the block is used.

## Time-expanded landing-zone representation
- For each hotspot cluster, maintain layers `(z, t)` where `z` is a landing UAV in the cluster and `t` ranges over retained time slots.
- Logical "stay" edges connect `(z, t)` to `(z, t+1)` whenever capacity is non-zero; "move" edges connect `(z, t)` to `(z_adjacent, t+1)` for neighbouring cells to model relocation between slots.
- In code we store this compactly via:
  - `capacity[z_index, t]` tables generated from `{B, I+}`.
  - Candidate time lists `cand_times[f_index][z_index]`.
  - Decision arrays `x[f, z, t]` and `u[f, z, t]` (or equivalent JuMP variables).
- This is equivalent to a time-expanded graph focused on ground bottlenecks; air-path nodes are implicit because inter-UAV links have no capacity constraints.

## Julia project structure
```
julia/
  Project.toml / Manifest.toml
  src/
    Main.jl              - CLI entry: parse JSON, orchestrate clustering & solves.
    IO.jl                - JSON schema for instances and schedules.
    Capacity.jl          - Generates per-cell capacity timeline from {B, I+}.
    Candidates.jl        - Builds time windows, prunes candidates.
    Clustering.jl        - Flow-zone graph, connected components.
    Model.jl             - JuMP model builder (variables, constraints, objective).
    Heuristics.jl        - Greedy warm-start respecting capacity & delay scoring.
    PostProcess.jl       - Assemble output format, score estimation.
  test/
    runtests.jl          - Unit tests for each module.
    data/*.json          - Tiny instances mirroring `examples/`.
```
- Keep the existing Python code in `src/` untouched. Add a Python helper (if needed) under `python_tools/` to export JSON instances for Julia.

## Solver and environment setup
- Julia >= 1.10.
- Install packages: `JuMP`, `HiGHS`, `JSON3`, `StructTypes`, `Graphs`, `DataStructures`. Add `Gurobi.jl` or `CPLEX.jl` if licenses are available; configure via environment variables.
- Enforce a global solver time limit (<= 5 s). For multiple clusters, allocate time proportionally to cluster size and stop once the global limit is reached.
- For nonlinear scoring terms (delay, distance), precompute coefficients so the JuMP model remains linear.

## Heuristics and warm starts
- Build a greedy schedule:
  1. For each flow, select the landing cell with maximal average capacity over the window and minimal travel time.
  2. Starting at `earliest_arrival`, assign bandwidth greedily while respecting per-slot capacity; move forward in time until the flow volume is exhausted, relocating to neighbours only when capacity drops to zero or a better option appears.
  3. Use the resulting `x`, `u`, and `y` values as a MIP start.
- Maintain feasibility even if the MIP time limit is hit so the solver always returns a usable schedule.

## Comparison with current Python solver
- Python code performs temporal scheduling with explicit path reasoning. The Julia MIP collapses the air path into Manhattan travel time while still handling per-second ground capacity exactly.
- Retain Python scripts for:
  - Instance parsing (reuse in Julia or export JSON).
  - Score verification against the official metric.
  - Visualization/debugging.

## Open questions & validation
- Confirm relocation assumptions: earliest-arrival rules currently come from Manhattan distance to each landing cell independently; add explicit adjacency constraints if empirical tests reveal that intra-zone movement needs extra travel time.
- Determine granularity for `x[f, z, t]` (integers vs. continuous). The official scoring works with real Mbps, so continuous variables are acceptable if the solver supports them.
- Validate the linear objective surrogate by scoring MIP solutions with the official formula on known examples.

## Implementation roadmap
1. **Define JSON schema** for inputs/outputs and export existing examples.
2. **Implement capacity timeline generator** using the 10-second profile.
3. **Implement clustering** and candidate generation with pruning.
4. **Code greedy warm-start** matching the challenge scoring.
5. **Build JuMP model** (variables, constraints, objective surrogate, time limits).
6. **Integrate solver loop** (per-cluster solve, time budgeting, incumbent handling).
7. **Validate** on provided examples; compare scores with Python reference.
8. **Document** setup steps and solver usage for the team.

## Deciding between Julia and Python
- Staying in Python would require a comparable MIP/MCF implementation (e.g., OR-Tools with CBC/SCIP) but may struggle to hit the 5-second target on larger clusters.
- Julia + JuMP offers cleaner modeling, first-class support for commercial solvers, and faster iterations. Recommendation: keep the existing Python solution as reference, build the new solver in Julia.
