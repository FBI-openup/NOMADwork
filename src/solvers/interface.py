from __future__ import annotations

from typing import Dict, List, Tuple

from ..model import Problem


def solve(problem: Problem) -> Dict[int, List[Tuple[int, int, int, float]]]:
    """Interface for solver backends.

    Returns schedules mapping: fid -> list of (t, x, y, z).
    """
    raise NotImplementedError

