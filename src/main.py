from __future__ import annotations

import argparse
import os
import sys

# Allow running as a script: python src/main.py
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.io import parse_stream, parse_file, format_output  # type: ignore
    from src.scheduler import GreedyScheduler  # type: ignore
else:
    from .io import parse_stream, parse_file, format_output
    from .scheduler import GreedyScheduler


def main() -> None:
    parser = argparse.ArgumentParser(description="UAV U2GL Scheduling Baseline")
    parser.add_argument("-i", "--input", help="Path to input file; if omitted, reads stdin")
    parser.add_argument("-o", "--output", help="Optional output file; default stdout")
    parser.add_argument("--solver", choices=["greedy", "milp"], default="greedy", help="Scheduler backend")
    parser.add_argument("--alpha", type=float, default=0.1, help="Distance penalty alpha")
    parser.add_argument("--top-k", type=int, default=30, help="Top-K landing candidates per flow")
    parser.add_argument("--stickiness", type=float, default=0.05, help="Stickiness bonus multiplier")
    parser.add_argument("--tmax-delay", type=int, default=10, help="T_max for delay weight")
    args = parser.parse_args()

    if args.input:
        problem = parse_file(args.input)
    else:
        problem = parse_stream(sys.stdin)

    if args.solver == "greedy":
        scheduler = GreedyScheduler(
            alpha=args.alpha, top_k=args.top_k, stickiness_bonus=args.stickiness, tmax_delay=args.tmax_delay
        )
        schedules = scheduler.schedule(problem)
    else:
        # MILP solver path (stubbed): try import and run
        try:
            if __package__ is None or __package__ == "":
                from src.solvers.milp_stub import solve_milp  # type: ignore
            else:
                from .solvers.milp_stub import solve_milp  # type: ignore
        except Exception as e:
            print("MILP solver backend not available: ", e, file=sys.stderr)
            sys.exit(2)
        schedules = solve_milp(problem)
    out_text = format_output(schedules)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_text)
    else:
        sys.stdout.write(out_text)


if __name__ == "__main__":
    main()
