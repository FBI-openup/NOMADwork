from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


Coord = Tuple[int, int]


@dataclass
class UAVNode:
    x: int
    y: int
    B: float  # peak bandwidth
    phase: int  # 0..9

    @property
    def coord(self) -> Coord:
        return (self.x, self.y)


@dataclass
class Flow:
    fid: int
    ax: int
    ay: int
    t_start: int
    total: float  # Mbits
    m1: int
    n1: int
    m2: int
    n2: int

    # runtime fields
    remaining: float = field(init=False)
    last_landing: Optional[Coord] = field(default=None, init=False)
    unique_landings: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.remaining = float(self.total)

    @property
    def access_coord(self) -> Coord:
        return (self.ax, self.ay)

    def landing_in_range(self, x: int, y: int) -> bool:
        return self.m1 <= x <= self.m2 and self.n1 <= y <= self.n2


@dataclass
class Problem:
    M: int
    N: int
    FN: int
    T: int
    uavs: Dict[Coord, UAVNode]
    flows: List[Flow]

    @property
    def grid_size(self) -> int:
        return self.M * self.N


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

