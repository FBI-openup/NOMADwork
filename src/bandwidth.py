from __future__ import annotations

from typing import Dict, Tuple

from .model import UAVNode, Coord


# Pattern over slots 0..9 as multipliers of B
PATTERN = [0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0]


def capacity_at(uav: UAVNode, t: int) -> float:
    idx = (uav.phase + t) % 10
    return uav.B * PATTERN[idx]


def capacities_for_t(uavs: Dict[Coord, UAVNode], t: int) -> Dict[Coord, float]:
    return {coord: capacity_at(uav, t) for coord, uav in uavs.items()}
