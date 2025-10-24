"""Greedy warm-start schedule respecting landing-zone rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .capacity import capacity_lookup
from .candidates import CandidateIndex, FlowCandidate
from .models import Flow, Problem, ScheduleSegment

EPSILON = 1e-6


@dataclass(slots=True)
class WarmStart:
    assignments: Dict[int, List[ScheduleSegment]]


def _score_landing_cell(flow: Flow, slots: List[int], cell: Tuple[int, int], capacity_table) -> float:
    total_capacity = sum(capacity_lookup(capacity_table, cell, time_slot) for time_slot in slots)
    travel = max(1, flow.travel_time(cell))
    return total_capacity / travel


def _ordered_cells(flow: Flow, candidate: FlowCandidate, capacity_table) -> List[Tuple[int, int]]:
    scored = [
        (cell, _score_landing_cell(flow, slots, cell, capacity_table))
        for cell, slots in candidate.time_slots.items()
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [cell for cell, _ in scored]


def build_warm_start(problem: Problem, index: CandidateIndex, capacity_table) -> WarmStart:
    remaining_capacity: Dict[Tuple[int, int], List[float]] = {}
    for cell, capacities in capacity_table.items():
        remaining_capacity[cell] = list(capacities)

    assignments: Dict[int, List[ScheduleSegment]] = {}
    flows_sorted = sorted(problem.flows, key=lambda flow: flow.start_time)

    for flow in flows_sorted:
        candidate = index.flows.get(flow.flow_id)
        if candidate is None:
            assignments[flow.flow_id] = []
            continue

        segments: List[ScheduleSegment] = []
        allocated = 0.0
        used_times: set[int] = set()

        for cell in _ordered_cells(flow, candidate, remaining_capacity):
            for time_slot in candidate.time_slots[cell]:
                if time_slot in used_times:
                    continue
                available = remaining_capacity[cell][time_slot]
                if available <= EPSILON:
                    continue
                remaining = flow.volume - allocated
                if remaining <= EPSILON:
                    break
                take = min(available, remaining)
                remaining_capacity[cell][time_slot] -= take
                segments.append(ScheduleSegment(time=time_slot, x=cell[0], y=cell[1], rate=take))
                used_times.add(time_slot)
                allocated += take
            if allocated + EPSILON >= flow.volume:
                break

        assignments[flow.flow_id] = segments

    return WarmStart(assignments=assignments)
