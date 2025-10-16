# NTN Paper Integration Notes

## Repository Snapshot
- `src/main.py` wires CLI options for the greedy and MILP schedulers, reading inputs via `src/io.py`.
- `src/scheduler.py` holds the baseline greedy heuristic that chooses a landing UAV per flow per second using per-slot capacity and a stickiness bonus.
- `src/solvers/milp_pulp.py` formulates the full horizon schedule as a MILP with binaries for per-slot landing selection and landing-count tracking.
- `src/scorer.py`, `src/bandwidth.py`, and `src/model.py` provide the scoring breakdown, periodic capacity pattern, and core dataclasses used by both solvers.
- `docs/brainstorm.md` captures initial design notes and future work ideas for the baseline; no temporal-graph or rolling-horizon method is implemented yet.

## NTN Paper Takeaways
- The paper models time-varying resources through a **time-expanded graph (TEG)** and augments it into an **extended TEG (ETEG)** with virtual nodes so heterogeneous resources can be scheduled jointly over time.
- Deterministic routing (DetR) jointly reasons about **link capacity and storage across cycles**, selecting hop-by-hop routes that stay feasible over time instead of relying on per-slot greedy picks.
- Their approach delivers near-ILP quality while keeping **polynomial time** by solving on sliding temporal layers and preferring stable paths that avoid churn between cycles.
- Compared to snapshot or static routing, the paper highlights the benefits of **look-ahead scheduling**, **cycle-level coordination**, and **landing/staging reuse** to mitigate micro-bursts and congestion.

## Improvement Opportunities
- Introduce a **receding-horizon planner** that builds a small TEG over the next `H` seconds, allocates transmissions jointly, and then commits only the immediate second—mirroring the DetR hop-by-hop selection.
- Incorporate an explicit **landing reuse incentive** so flows keep using the same ground station when possible, reducing the effective landing-count penalty (the 1/k term).
- Leverage the TEG window to estimate **future contention** and delay benefit, letting the scheduler defer non-urgent flows toward higher-value future slots instead of exhausting current capacity greedily.
- Expose the temporal scheduler via the CLI (e.g., `--solver temporal`) and document how it derives from the NTN deterministic routing ideas, differentiating it from the existing greedy and MILP endpoints.
