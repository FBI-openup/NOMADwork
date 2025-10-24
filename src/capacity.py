"""Capacity helpers for per-UAV time-varying bandwidth."""

from __future__ import annotations

from typing import Dict, Tuple

from .models import Problem, UAV

CAPACITY_PATTERN = (0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0)


def capacity_at(uav: UAV, time_slot: int) -> float:
    index = (uav.phase + time_slot) % len(CAPACITY_PATTERN)
    return uav.peak_bandwidth * CAPACITY_PATTERN[index]


def build_capacity_table(problem: Problem) -> Dict[Tuple[int, int], Tuple[float, ...]]:
    horizon = problem.grid.horizon
    table: Dict[Tuple[int, int], Tuple[float, ...]] = {}
    for uav in problem.uavs:
        capacity_row = tuple(capacity_at(uav, time_slot) for time_slot in range(horizon))
        table[(uav.x, uav.y)] = capacity_row
    return table


def capacity_lookup(
    capacity_table: Dict[Tuple[int, int], Tuple[float, ...]],
    cell: Tuple[int, int],
    time_slot: int,
) -> float:
    return capacity_table[cell][time_slot]
