from __future__ import annotations

import io

from src.io import parse_file, format_output
from src.scheduler import GreedyScheduler
from src.scorer import score
from src.solvers.milp_pulp import solve_milp


def test_example_greedy_scores_high():
    problem = parse_file("examples/example1.in")
    schedules = GreedyScheduler().schedule(problem)
    s = score(problem, schedules)
    assert s >= 90.0


def test_example_milp_scores_high():
    problem = parse_file("examples/example1.in")
    schedules = solve_milp(problem, top_k=9)
    s = score(problem, schedules)
    assert s >= 90.0

