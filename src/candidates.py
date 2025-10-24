"""Candidate time-slot generation for each flow and landing cell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .capacity import capacity_lookup
from .models import Flow, Problem


@dataclass(slots=True)
class CandidateConfig:
    base_window: int = 20
    expand_step: int = 10
    max_window: int = 200
    min_capacity: float = 1e-6


@dataclass(slots=True)
class FlowCandidate:
    flow: Flow
    time_slots: Dict[Tuple[int, int], List[int]]


@dataclass(slots=True)
class CandidateIndex:
    flows: Dict[int, FlowCandidate]


def _collect_time_slots(
    flow: Flow,
    landing_cell: Tuple[int, int],
    horizon: int,
    capacity_table,
    config: CandidateConfig,
) -> List[int]:
    earliest_arrival = flow.earliest_arrival(landing_cell)
    if earliest_arrival >= horizon:
        return []

    selected: List[int] = []
    seen = set()
    total_capacity = 0.0
    window = config.base_window

    def consider(until: int) -> float:
        nonlocal total_capacity
        for time_slot in range(earliest_arrival, min(until, horizon - 1) + 1):
            if time_slot in seen:
                continue
            seen.add(time_slot)
            capacity = capacity_lookup(capacity_table, landing_cell, time_slot)
            if capacity > config.min_capacity:
                selected.append(time_slot)
                total_capacity += capacity
        return total_capacity

    consider(earliest_arrival + window)
    while (
        total_capacity + config.min_capacity < flow.volume
        and window < config.max_window
        and earliest_arrival + window < horizon
    ):
        window = min(config.max_window, window + config.expand_step)
        consider(earliest_arrival + window)

    selected.sort()
    return selected


def build_candidate_index(problem: Problem, capacity_table, config: CandidateConfig) -> CandidateIndex:
    flow_map: Dict[int, FlowCandidate] = {}
    for flow in problem.flows:
        slot_map: Dict[Tuple[int, int], List[int]] = {}
        for landing_cell in flow.zone.cells():
            slots = _collect_time_slots(flow, landing_cell, problem.grid.horizon, capacity_table, config)
            if slots:
                slot_map[landing_cell] = slots
        flow_map[flow.flow_id] = FlowCandidate(flow=flow, time_slots=slot_map)
    return CandidateIndex(flows=flow_map)
