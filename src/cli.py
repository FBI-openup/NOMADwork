"""Command-line interface for the Python solver."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .candidates import CandidateConfig
from .parser import load_problem, ProblemFormatError
from .postprocess import assignments_to_flow_list, schedule_to_dict, schedule_to_text
from .solver import SolverConfig, solve_problem
from .validation import ScheduleValidationError, validate_schedule


def _score_summary_to_dict(summary):
    return {
        "total": summary.total,
        "by_flow": [
            {
                "flow_id": item.flow_id,
                "traffic": item.traffic,
                "delay": item.delay,
                "distance": item.distance,
                "landing": item.landing,
                "weight": item.weight,
            }
            for item in summary.by_flow
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Huawei traffic allocation solver (Python version)")
    parser.add_argument("--input", "-i", help="Path to .in or JSON problem file (reads stdin when omitted)")
    parser.add_argument("--output-text", help="Optional file to write the canonical .out schedule")
    parser.add_argument("--output-json", help="Optional file to write assignments and score JSON")
    parser.add_argument("--time-limit", type=float, help="Override the global solver time budget (seconds)")
    parser.add_argument("--validate", action="store_true", help="Validate the produced schedule")
    args = parser.parse_args(argv)

    try:
        if args.input:
            problem = load_problem(args.input)
        else:
            raw_text = sys.stdin.read()
            from .parser import parse_problem_from_text

            problem = parse_problem_from_text(raw_text)
    except ProblemFormatError as exc:
        print(f"Failed to parse problem: {exc}", file=sys.stderr)
        return 2

    solver_config = SolverConfig()
    if args.time_limit is not None:
        solver_config = SolverConfig(total_time_limit=args.time_limit)

    result = solve_problem(problem, solver_config=solver_config, candidate_config=CandidateConfig())
    flow_assignments = assignments_to_flow_list(result.assignments)

    if args.validate:
        try:
            validate_schedule(problem, result.assignments)
        except ScheduleValidationError as exc:
            print(f"Schedule validation failed: {exc}", file=sys.stderr)
            return 3

    schedule_text = schedule_to_text(flow_assignments)
    if args.output_text:
        Path(args.output_text).write_text(schedule_text, encoding="utf-8")
    else:
        print(schedule_text)

    if args.output_json:
        payload = schedule_to_dict(flow_assignments)
        payload["score"] = _score_summary_to_dict(result.score)
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"Solved {len(problem.flows)} flows | score={result.score.total:.3f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
