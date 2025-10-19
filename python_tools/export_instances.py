"""Export challenge .in files to the JSON format consumed by the Julia solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from src.io import parse_file


def _problem_to_dict(problem) -> dict:
    """Convert the Python Problem dataclass into the JSON schema expected by Julia."""
    uavs = [
        {
            "x": uav.x,
            "y": uav.y,
            "peak_bandwidth": uav.B,
            "phase": uav.phase,
        }
        for uav in problem.uavs.values()
    ]
    flows = [
        {
            "id": flow.fid,
            "access": {"x": flow.ax, "y": flow.ay},
            "start_time": flow.t_start,
            "volume": flow.total,
            "zone": {
                "x_min": flow.m1,
                "y_min": flow.n1,
                "x_max": flow.m2,
                "y_max": flow.n2,
            },
        }
        for flow in problem.flows
    ]
    return {
        "grid": {"rows": problem.M, "cols": problem.N, "horizon": problem.T},
        "flows": flows,
        "uavs": uavs,
    }


def export_instance(path: Path, output_path: Path, indent: Optional[int] = 2) -> None:
    problem = parse_file(str(path))
    payload = _problem_to_dict(problem)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent if indent is not None else None)
        if indent is not None:
            f.write("\n")


def resolve_output_path(input_path: Path, output_dir: Optional[Path]) -> Path:
    if output_dir is None:
        return input_path.with_suffix(".json")
    inferred = input_path.with_suffix(".json").name
    return output_dir / inferred


def export_many(paths: Iterable[Path], output_dir: Optional[Path], indent: Optional[int]) -> None:
    for path in paths:
        output_path = resolve_output_path(path, output_dir)
        export_instance(path, output_path, indent=indent)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Path(s) to .in files to convert.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store converted JSON files (defaults to alongside input).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent width for JSON output (set to 0 for compact output).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    indent: Optional[int]
    if args.indent <= 0:
        indent = None
    else:
        indent = args.indent
    export_many(args.inputs, args.output_dir, indent)


if __name__ == "__main__":
    main()
