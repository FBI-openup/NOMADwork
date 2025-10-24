"""Core data structures shared by the solver modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


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
class AccessPoint:
    x: int
    y: int


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
    access: AccessPoint
    start_time: int
    volume: float
    zone: LandingZone

    def travel_time(self, landing_cell: Tuple[int, int]) -> int:
        return abs(self.access.x - landing_cell[0]) + abs(self.access.y - landing_cell[1])

    def earliest_arrival(self, landing_cell: Tuple[int, int]) -> int:
        return self.start_time + self.travel_time(landing_cell)


@dataclass(slots=True)
class Problem:
    grid: Grid
    uavs: List[UAV]
    flows: List[Flow]

    def uav_lookup(self) -> Dict[Tuple[int, int], UAV]:
        return {(uav.x, uav.y): uav for uav in self.uavs}


@dataclass(slots=True)
class ScheduleSegment:
    time: int
    x: int
    y: int
    rate: float


@dataclass(slots=True)
class FlowAssignment:
    flow_id: int
    segments: List[ScheduleSegment] = field(default_factory=list)


@dataclass(slots=True)
class ScoreSlice:
    flow_id: int
    traffic: float
    delay: float
    distance: float
    landing: float
    weight: float


@dataclass(slots=True)
class ScoreSummary:
    total: float
    by_flow: List[ScoreSlice]
