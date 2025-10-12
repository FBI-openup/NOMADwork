from __future__ import annotations

from typing import Dict, List, Tuple

from ..model import Problem


def solve_milp(problem: Problem) -> Dict[int, List[Tuple[int, int, int, float]]]:
    msg = (
        "MILP solver not implemented yet. See docs/solver_plan.md for the plan. "
        "Recommend using Gurobi/CPLEX/SCIP with SOS1, top-K pruning, and a warm start from the greedy schedule."
    )
    raise RuntimeError(msg)

