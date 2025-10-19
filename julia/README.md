# Julia Solver

This package hosts the JuMP-based solver described in `next_steps.md`. It complements the existing Python pipeline without replacing it; the Python code remains the authoritative reference for input parsing, scoring, and benchmarking.

## Environment setup
1. Install Julia ≥ 1.10.
2. From the repository root run:
   ```sh
   julia --project=julia -e 'using Pkg; Pkg.instantiate()'
   ```
   This pulls `JuMP`, `HiGHS`, `JSON3`, `StructTypes`, `Graphs`, and `DataStructures`.
3. (Optional) Install a commercial MIP solver such as Gurobi and configure it via the usual environment variables. The default configuration uses `HiGHS`.

## Data interchange
- Use `python python_tools/export_instances.py <path/to/input.in> -o julia/test/data` to convert legacy `.in` files into the JSON schema documented in `docs/julia_json_schema.md`.
- The Julia CLI can emit both JSON payloads and the standard `.out` schedule format, enabling round-trips with the existing Python scorer (`python -m src.scorer`).

## Running the solver
```sh
# JSON file input, write .out file
julia --project=julia -e "using NTNScheduler; NTNScheduler.run_cli(ARGS)" -- --input julia/test/data/example1.json --output-text example1_julia.out

# Native .in input, emit schedule to stdout
julia --project=julia -e "using NTNScheduler; NTNScheduler.run_cli(ARGS)" -- --input examples/example1.in

# Native .in input, write .out file
julia --project=julia -e "using NTNScheduler; NTNScheduler.run_cli(ARGS)" -- --input examples/gen1.in --output-text examples/julia_gen1.out

# Stream .in over stdin and capture stdout
cat examples/example1.in | julia --project=julia -e "using NTNScheduler; NTNScheduler.run_cli(String[])" > example1.out
```
Key flags / behaviours:
- `--input` accepts either `.json` or native `.in` files. If omitted, the solver reads from standard input.
- `--output-text` writes the canonical `.out` submission; when omitted the schedule is printed to standard output.
- `--output-json` writes the richer JSON payload (assignments + score breakdown).
- `--time-limit` overrides the default 5 second global budget shared across clusters.

## Testing
Execute the automated checks with:
```sh
julia --project=julia -e 'using Pkg; Pkg.test()'
```
The test suite covers capacity pattern generation, candidate pruning, warm-start feasibility, score replication against the Python reference, and an end-to-end smoke test on `example1.json`.
