"""CLI wrapper to score and validate existing schedules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .parser import load_problem, ProblemFormatError
from .postprocess import assignments_to_flow_list, schedule_to_dict
from .scorer import ScheduleFormatError, load_schedule, score_schedule
from .validation import ScheduleValidationError


def _summary_to_dict(score):
    return {
        "total": score.total,
        "by_flow": [
            {
                "flow_id": item.flow_id,
                "traffic": item.traffic,
                "delay": item.delay,
                "distance": item.distance,
                "landing": item.landing,
                "weight": item.weight,
            }
            for item in score.by_flow
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score and validate an existing .out schedule.")
    parser.add_argument("--input", "-i", required=True, help="Path to the problem definition (.in or JSON)")
    parser.add_argument("--schedule", "-s", required=True, help="Path to the schedule file (.out)")
    parser.add_argument("--validate", action="store_true", help="Run organiser rule validation")
    parser.add_argument("--output-json", help="Optional path to write score + assignments JSON")
    args = parser.parse_args(argv)

    try:
        problem = load_problem(args.input)
    except ProblemFormatError as exc:
        print(f"Failed to parse problem: {exc}", file=sys.stderr)
        return 2

    try:
        assignments = load_schedule(args.schedule)
    except ScheduleFormatError as exc:
        print(f"Failed to parse schedule: {exc}", file=sys.stderr)
        return 2

    try:
        score = score_schedule(problem, assignments, validate=args.validate)
    except ScheduleValidationError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 3

    print(f"Score: {score.total:.3f}")

    if args.output_json:
        flow_assignments = assignments_to_flow_list(assignments)
        payload = schedule_to_dict(flow_assignments)
        payload["score"] = _summary_to_dict(score)
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
