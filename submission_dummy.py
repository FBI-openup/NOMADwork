#!/usr/bin/env python3
"""
Dummy submission to verify solver availability.

Behaviour:
- Checks that PuLP is installed and that either HiGHS or the default CBC solver
  is available. If no solver backend is reachable, exits with a RuntimeError.
- If a solver is available, constructs a very simple greedy schedule that
  respects capacity/time/landing constraints. This is intentionally lightweight
  and intended only for diagnostics.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

try:
    import pulp
except ModuleNotFoundError as exc:  # pragma: no cover - diagnostic behaviour
    raise RuntimeError("PuLP is required for submission_dummy.py") from exc


# ---------------------------------------------------------------------------
# Basic data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Grid:
    rows: int
    cols: int
    horizon: int


@dataclass(slots=True)
class UAV:
    x: int
    y: int
    peak_bandwidth: float
    phase: int


@dataclass(slots=True)
class LandingZone:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def cells(self) -> Iterable[Tuple[int, int]]:
        for x in range(self.x_min, self.x_max + 1):
            for y in range(self.y_min, self.y_max + 1):
                yield (x, y)


@dataclass(slots=True)
class Flow:
    flow_id: int
    access_x: int
    access_y: int
    start_time: int
    volume: float
    zone: LandingZone

    def travel_time(self, cell: Tuple[int, int]) -> int:
        return abs(self.access_x - cell[0]) + abs(self.access_y - cell[1])

    def earliest_arrival(self, cell: Tuple[int, int]) -> int:
        return self.start_time + self.travel_time(cell)


@dataclass(slots=True)
class Problem:
    grid: Grid
    uavs: List[UAV]
    flows: List[Flow]


@dataclass(slots=True)
class ScheduleSegment:
    time: int
    x: int
    y: int
    rate: float


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_tokens(tokens: List[str]) -> Problem:
    idx = 0

    def next_token() -> str:
        nonlocal idx
        if idx >= len(tokens):
            raise ValueError("unexpected end of input")
        token = tokens[idx]
        idx += 1
        return token

    rows = int(next_token())
    cols = int(next_token())
    flow_count = int(next_token())
    horizon = int(next_token())

    uavs: List[UAV] = []
    for _ in range(rows * cols):
        x = int(next_token())
        y = int(next_token())
        peak_bandwidth = float(next_token())
        phase = int(next_token())
        uavs.append(UAV(x=x, y=y, peak_bandwidth=peak_bandwidth, phase=phase))

    flows: List[Flow] = []
    for _ in range(flow_count):
        fid = int(next_token())
        ax = int(next_token())
        ay = int(next_token())
        start = int(next_token())
        volume = float(next_token())
        m1 = int(next_token())
        n1 = int(next_token())
        m2 = int(next_token())
        n2 = int(next_token())
        flows.append(
            Flow(
                flow_id=fid,
                access_x=ax,
                access_y=ay,
                start_time=start,
                volume=volume,
                zone=LandingZone(x_min=m1, y_min=n1, x_max=m2, y_max=n2),
            )
        )

    return Problem(grid=Grid(rows=rows, cols=cols, horizon=horizon), uavs=uavs, flows=flows)


def parse_problem(raw: str) -> Problem:
    stripped = raw.lstrip()
    if not stripped:
        raise ValueError("input is empty")
    if stripped[0] in "{[":
        import json

        payload = json.loads(raw)
        grid = payload["grid"]
        uavs = [
            UAV(
                x=int(item["x"]),
                y=int(item["y"]),
                peak_bandwidth=float(item["peak_bandwidth"]),
                phase=int(item["phase"]),
            )
            for item in payload["uavs"]
        ]
        flows = [
            Flow(
                flow_id=int(item["id"]),
                access_x=int(item["access"]["x"]),
                access_y=int(item["access"]["y"]),
                start_time=int(item["start_time"]),
                volume=float(item["volume"]),
                zone=LandingZone(
                    x_min=int(item["zone"]["x_min"]),
                    y_min=int(item["zone"]["y_min"]),
                    x_max=int(item["zone"]["x_max"]),
                    y_max=int(item["zone"]["y_max"]),
                ),
            )
            for item in payload["flows"]
        ]
        return Problem(
            grid=Grid(rows=int(grid["rows"]), cols=int(grid["cols"]), horizon=int(grid["horizon"])),
            uavs=uavs,
            flows=flows,
        )
    return _parse_tokens(raw.split())


# ---------------------------------------------------------------------------
# Capacity handling
# ---------------------------------------------------------------------------


CAPACITY_PATTERN = (0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0)


def capacity_at(uav: UAV, t: int) -> float:
    idx = (uav.phase + t) % len(CAPACITY_PATTERN)
    return uav.peak_bandwidth * CAPACITY_PATTERN[idx]


def build_capacity_table(problem: Problem) -> Dict[Tuple[int, int], List[float]]:
    table: Dict[Tuple[int, int], List[float]] = {}
    for uav in problem.uavs:
        table[(uav.x, uav.y)] = [capacity_at(uav, t) for t in range(problem.grid.horizon)]
    return table


# ---------------------------------------------------------------------------
# Dummy greedy solution (fast, respects constraints)
# ---------------------------------------------------------------------------


def solve_greedy(problem: Problem) -> Dict[int, List[ScheduleSegment]]:
    table = build_capacity_table(problem)
    remaining = {cell: caps.copy() for cell, caps in table.items()}
    assignments: Dict[int, List[ScheduleSegment]] = {}

    for flow in sorted(problem.flows, key=lambda f: f.start_time):
        segments: List[ScheduleSegment] = []
        needed = flow.volume
        used_per_time: Dict[int, Tuple[int, int]] = {}

        for cell in flow.zone.cells():
            if needed <= 1e-6:
                break
            if cell not in table:
                continue
            earliest = flow.earliest_arrival(cell)
            cap_list = remaining[cell]
            for t in range(earliest, problem.grid.horizon):
                if needed <= 1e-6:
                    break
                if used_per_time.get(t):
                    continue  # already landed elsewhere this second
                available = cap_list[t]
                if available <= 1e-6:
                    continue
                take = min(available, needed)
                cap_list[t] -= take
                needed -= take
                used_per_time[t] = cell
                segments.append(ScheduleSegment(time=t, x=cell[0], y=cell[1], rate=take))

        segments.sort(key=lambda seg: (seg.time, seg.x, seg.y))
        assignments[flow.flow_id] = segments

    return assignments


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def schedule_to_text(problem: Problem, assignments: Dict[int, List[ScheduleSegment]]) -> str:
    lines: List[str] = []
    for flow in problem.flows:
        segs = assignments.get(flow.flow_id, [])
        lines.append(f"{flow.flow_id} {len(segs)}")
        for seg in segs:
            lines.append(f"{seg.time} {seg.x} {seg.y} {seg.rate:.6f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def check_solver_available() -> None:
    """Ensure at least one MILP solver backend is available."""
    solver = None
    if hasattr(pulp, "HiGHS_CMD"):
        solver = pulp.HiGHS_CMD(msg=False)
    else:
        solver = pulp.PULP_CBC_CMD(msg=False)

    try:
        available = solver.available()
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError("Unable to determine solver availability") from exc

    if not available:
        raise RuntimeError("Required PuLP solver backend is unavailable.")


def main() -> None:
    check_solver_available()
    raw = sys.stdin.read()
    if not raw.strip():
        return
    problem = parse_problem(raw)
    assignments = solve_greedy(problem)
    sys.stdout.write(schedule_to_text(problem, assignments))


if __name__ == "__main__":
    main()
