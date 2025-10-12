# Brainstorm, Assumptions, and Plan

## Problem Understanding

We schedule per‑second traffic for multiple flows in an MxN UAV grid. Each flow must choose at most one landing UAV per time slot within its allowed rectangle, respecting per‑UAV U2GL bandwidth that varies periodically with per‑UAV phase. Objective combines: completeness, delay, distance, and number of distinct landing UAVs used.

## Key Observations

- U2GL is the bottleneck; mesh is implicit via the distance penalty (Manhattan hops).
- Bandwidth pattern is deterministic and periodic; average capacity is proportional to B (independent of phase), while instantaneous capacity depends on slot and phase.
- Good schedules preferentially send early, over short distances, using fewer unique landing UAVs, without exceeding U2GL capacities.

## Baseline Heuristic

Per time slot t:

- Compute effective capacity for every UAV given (B, phase).
- Build the active flow set (started and not completed).
- For each flow, precompute top‑K landing candidates by static score `B * 2^{-alpha * distance}`; restrict per‑slot search to those K to keep runtime manageable.
- For the dynamic choice, use utility:
  `util = capacity(x',y',t) * 2^{-alpha * distance} + stickiness_bonus`
  where `stickiness_bonus` is applied if the chosen landing equals the last used landing UAV.
- Greedily iterate flows by priority (earlier start, larger remaining, higher delay weight) and allocate `z = min(remaining_flow, capacity_at_landing)` for the second.

This ensures feasibility and captures the main drivers of the score.

## Complexity Considerations

- Grid up to 70x70 (<= 4900 UAVs), T <= 500, FN < 5000.
- Computing all UAV capacities each second is O(MN) and cheap.
- Limiting landing search per flow to K (default 30) keeps per‑second work ~ O(FN*K + MN).

## Potential Improvements

- Per‑UAV fair sharing or small linear program per time step to improve allocation when many flows collide on a few strong U2GLs.
- Landing point change minimization via explicit penalties or a budgeted change counter.
- Multi‑step lookahead (rolling horizon) to align flow bursts to future peaks.
- Candidate set augmentation when top‑K have zero capacity at t (fallback to next best in rectangle).
- Adaptive K by rectangle size and contention metrics.

## Plan

1. Implement core models and I/O (done).
2. Implement bandwidth model (period, phase).
3. Implement greedy baseline scheduler with top‑K candidates and stickiness.
4. Wire CLI and output formatting.
5. Implement scorer and example to enable local validation.
6. Document usage, assumptions, and next steps to ease iteration.

