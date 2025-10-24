## Objectives
- Replace the Julia solver with a Python implementation under a new `src/` package while satisfying the challenge specs in `problem_definition.md` and the clarifications in `Q_and_A.txt`.
- Preserve the useful modelling ideas (per-slot capacity tables, candidate pruning, clustering of overlapping landing zones, MILP formulation, scoring) but re-express them in readable Python suitable for beginner-to-intermediate maintainers.
- Enforce the “Manhattan spawn” rule: flows may only start transmitting after travelling the Manhattan distance to a landing-zone UAV, and they must remain inside their allowed rectangle with at most one active landing UAV per time slot.
- Provide clear legality checks and tests that reject any schedule violating organiser rules (single landing UAV per slot, staying inside the zone, respect for release times and capacity).

## Keep From The Julia Implementation
- Input semantics: grid/flow data parsing that supports both `.in` and JSON payloads; replicate the token-by-token parsing logic in Python dataclasses.
- Capacity modelling: reuse the 10-slot periodic profile from `Capacity.jl` and precompute `(x, y, t) → capacity` tables to avoid recomputation.
- Candidate generation: keep the expanding time-window heuristic with per-cell pruning based on earliest arrival and capacity availability.
- Clustering: solve flow groups independently when their landing rectangles overlap, mirroring `Clustering.jl` so superposed landing zones are handled together and disjoint ones separately.
- Warm-start heuristic: adapt the greedy single-cell filler so the PuLP model has a feasible fallback schedule even if the MILP fails or times out.
- MILP structure: carry over the `x[f,z,t]`, `u[f,z,t]`, `y[f,z]`, and `drop[f]` variables, the capacity/one-cell-per-slot/landing-activation constraints, and the linearised scoring surrogate.
- Post-processing: reuse the scoring breakdown (`traffic`, `delay`, `distance`, `landing`) and the text output formatting so the new solver remains comparable with historical results.

## Rebuild Or Adjust In Python
- Discard Julia-only plumbing (`NTNScheduler` module wiring, StructTypes integration, CLI helpers) and reimplement them as plain Python modules with explicit, readable function names.
- Replace Julia’s implicit multiple dispatch with Python `@dataclass` models and typed helper functions; avoid short variable names (e.g., favour `landing_cell` over `lc`).
- Revisit the solver configuration surface so PuLP/CBC options (time limits, mip gap, penalties) are exposed via a simple `SolverConfig` dataclass.
- Implement a standalone `validation` module that codifies the organiser rules, including “one landing UAV per flow per time slot,” “no leaving the landing zone,” “no transmission before earliest arrival,” and capacity checks, raising descriptive errors for illegal schedules.
- Ensure JSON I/O mirrors the documented schema but add docstrings and comments for maintainers unfamiliar with the challenge.
- Remove reliance on Julia’s warm-start API; instead, translate the heuristic output into PuLP variable start hints when supported, or use it as a fallback solution when CBC cannot be nudged.

## Proposed `src/` Layout (initial draft)
- `src/models.py` – dataclasses for `Grid`, `UAV`, `Flow`, `LandingZone`, `Problem`, and schedule primitives.
- `src/parser.py` – `.in` and JSON readers producing `Problem` instances, plus command-line argument handling for input discovery.
- `src/capacity.py` – capacity pattern constants, `capacity_at`, and timeline builders.
- `src/candidates.py` – earliest-arrival computation, candidate slot generation, and configuration knobs.
- `src/clustering.py` – discovery of overlapping flow groups based on landing rectangles/candidate cells.
- `src/heuristics.py` – greedy warm-start builder returning per-flow schedule segments.
- `src/solver.py` – PuLP-based MILP with variables/constraints mirroring the Julia logic and support for cluster-by-cluster solves within a global time budget.
- `src/postprocess.py` – schedule merging, normalisation, scoring, and text/JSON emitters.
- `src/validation.py` – rule checks driven by organiser clarifications, leveraged both in tests and as an optional CLI flag.
- `src/cli.py` – user-facing entry point wiring everything together (`parse → solve → validate → output`).
- `src/__init__.py` – expose key functions (`solve_problem`, `validate_schedule`, etc.) for external callers.

## Migration Steps
1. Scaffold the `src/` package with the module skeletons above, including docstrings that restate the relevant problem rules.
2. Port data models and parsing logic from `IO.jl`, translating the structs to Python dataclasses and ensuring both `.in` and JSON formats round-trip with existing example files.
3. Implement capacity helpers identical to `Capacity.jl`, backed by unit tests that match the known pattern (e.g., slots 0/1/8/9 = 0, slots 3–6 = B).
4. Translate the candidate generation process, keeping the expanding window/min-capacity rules and guaranteeing that earliest-arrival equals start time + Manhattan distance.
5. Recreate flow clustering using Python sets/queues, verifying that overlapping landing zones fall into the same component while disjoint zones do not.
6. Build the greedy warm-start to produce schedules that respect capacity and single-cell-per-time constraints; store it for fallback use.
7. Reimplement the MILP in PuLP, porting each Julia constraint and the weighted objective, and add hooks for solver limits, penalties, and optional warm-start hints.
8. Port the scoring and output assembly logic, validating it against historical `.out` files to ensure identical totals and per-flow slices.
9. Develop the validation module to enforce all organiser rules, integrating checks into both the solver pipeline (post-solve) and standalone tests.
10. Assemble the CLI around these pieces, supporting the same flags as the Julia version (`--input`, `--output-text`, `--output-json`, `--time-limit`) plus an optional `--validate` switch.
11. Remove or archive Julia-specific runtime docs (e.g., note in `README.md` that the Python solver supersedes the Julia one) once the Python path is proven.

## Testing And Validation Plan
- Use `pytest` for the new suite; port Julia’s test coverage (parsing, capacity, candidate generation, warm-start feasibility, scoring) into Python counterparts.
- Add validator-focused tests that intentionally build illegal schedules: two landing UAVs in the same time slot, transmissions outside the zone, and pre-arrival transmissions — each must raise a descriptive exception referencing the relevant rule.
- Re-score historical solutions (`examples/example1.out`, `examples/gen1_milp.out`) to ensure the Python scorer matches published totals to within a tight tolerance.
- Exercise the full solver on `examples/example1.in` and `examples/gen1.in`, asserting that the validator passes and that the score meets or exceeds the prior Julia baseline.
- Integrate the validator into CI by default so regressions in legality rules fail fast.

## Maintainability Guidelines
- Prefer expressive variable and function names; reserve abbreviations for standard terms (`uav`, `json`) and document any domain-specific terminology in module docstrings.
- Keep modules under ~300 lines where practical and factor reusable helpers out of monolithic files.
- Document solver configuration defaults in `SolverConfig` and surface them via CLI help text.
- Inline only short comments; create Markdown docs under `docs/` for deeper rationale (e.g., how the objective approximates the official score).
