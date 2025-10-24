"""Legality checks for schedules according to organiser rules."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from .capacity import build_capacity_table, capacity_lookup
from .models import Flow, Problem, ScheduleSegment


class ScheduleValidationError(ValueError):
    """Raised when a schedule violates competition rules."""


def _group_segments_by_flow(assignments: Dict[int, List[ScheduleSegment]]) -> Dict[int, List[ScheduleSegment]]:
    return {flow_id: list(segments) for flow_id, segments in assignments.items()}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScheduleValidationError(message)


def validate_schedule(problem: Problem, assignments: Dict[int, List[ScheduleSegment]], capacity_table=None) -> None:
    """Validate organiser rules and raise `ScheduleValidationError` on failure."""
    if capacity_table is None:
        capacity_table = build_capacity_table(problem)

    flow_map = {flow.flow_id: flow for flow in problem.flows}
    horizon = problem.grid.horizon

    # Check per-flow constraints.
    for flow_id, segments in assignments.items():
        flow = flow_map.get(flow_id)
        if flow is None:
            raise ScheduleValidationError(f"schedule references unknown flow {flow_id}")

        per_time: Dict[int, List[ScheduleSegment]] = {}
        for segment in segments:
            _require(segment.rate >= 0.0, f"flow {flow_id} has negative rate at time {segment.time}")
            _require(0 <= segment.time < horizon, f"flow {flow_id} uses invalid time slot {segment.time}")
            per_time.setdefault(segment.time, []).append(segment)

            _require(
                flow.zone.x_min <= segment.x <= flow.zone.x_max
                and flow.zone.y_min <= segment.y <= flow.zone.y_max,
                f"flow {flow_id} uses landing cell ({segment.x}, {segment.y}) outside its zone",
            )
            earliest = flow.earliest_arrival((segment.x, segment.y))
            _require(
                segment.time >= earliest,
                f"flow {flow_id} transmits at time {segment.time} before earliest arrival {earliest}",
            )

        for time_slot, group in per_time.items():
            _require(
                len(group) == 1,
                f"flow {flow_id} uses multiple landing cells at time {time_slot}",
            )

    # Check per-UAV capacity.
    usage: Dict[Tuple[int, int, int], float] = {}
    for flow_id, segments in assignments.items():
        for segment in segments:
            key = (segment.x, segment.y, segment.time)
            usage[key] = usage.get(key, 0.0) + segment.rate

    for (x, y, time_slot), total in usage.items():
        available = capacity_lookup(capacity_table, (x, y), time_slot)
        _require(
            total <= available + 1e-6,
            f"capacity exceeded at cell ({x}, {y}) time {time_slot}: used {total}, available {available}",
        )
