from __future__ import annotations

import argparse
import os
import sys

# Allow running as a script: python src/main.py
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.io import parse_stream, parse_file, format_output  # type: ignore
    from src.scheduler import GreedyScheduler  # type: ignore
    from src.scheduler_temporal import TemporalGraphScheduler  # type: ignore
else:
    from .io import parse_stream, parse_file, format_output
    from .scheduler import GreedyScheduler
    from .scheduler_temporal import TemporalGraphScheduler


def main() -> None:
    parser = argparse.ArgumentParser(description="UAV U2GL Scheduling Baseline")
    parser.add_argument("-i", "--input", help="Path to input file; if omitted, reads stdin")
    parser.add_argument("-o", "--output", help="Optional output file; default stdout")
    parser.add_argument(
        "--solver",
        choices=["greedy", "temporal", "milp"],
        default="greedy",
        help="Scheduler backend",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Distance penalty alpha")
    parser.add_argument("--top-k", type=int, default=30, help="Top-K landing candidates per flow")
    parser.add_argument("--stickiness", type=float, default=0.05, help="Stickiness bonus multiplier")
    parser.add_argument("--tmax-delay", type=int, default=10, help="T_max for delay weight")
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Lookahead horizon (seconds) for the temporal scheduler",
    )
    parser.add_argument(
        "--temporal-reuse",
        type=float,
        default=0.04,
        help="Reuse bonus applied when the temporal scheduler keeps using an existing landing",
    )
    parser.add_argument(
        "--temporal-penalty",
        type=float,
        default=0.06,
        help="Penalty applied when the temporal scheduler opens a new landing",
    )
    args = parser.parse_args()

    if args.input:
        problem = parse_file(args.input)
    else:
        problem = parse_stream(sys.stdin)

    if args.solver == "greedy":
        print("Using Greedy scheduler")
        scheduler = GreedyScheduler(
            alpha=args.alpha, top_k=args.top_k, stickiness_bonus=args.stickiness, tmax_delay=args.tmax_delay
        )
        schedules = scheduler.schedule(problem)
    elif args.solver == "temporal":
        print("Using Temporal graph scheduler")
        scheduler = TemporalGraphScheduler(
            alpha=args.alpha,
            top_k=args.top_k,
            horizon=args.horizon,
            stickiness_bonus=args.stickiness,
            reuse_bonus=args.temporal_reuse,
            new_landing_penalty=args.temporal_penalty,
            tmax_delay=args.tmax_delay,
        )
        schedules = scheduler.schedule(problem)
    else:
        # MILP solver path: try import and run
        try:
            if __package__ is None or __package__ == "":
                from src.solvers.milp_pulp import solve_milp  # type: ignore
            else:
                from .solvers.milp_pulp import solve_milp  # type: ignore
        except Exception as e:
            print("MILP solver backend not available: ", e, file=sys.stderr)
            sys.exit(2)
        schedules = solve_milp(problem, alpha=args.alpha, top_k=args.top_k, tmax_delay=args.tmax_delay)
    out_text = format_output(schedules)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_text)
    else:
        sys.stdout.write(out_text)


if __name__ == "__main__":
    main()
