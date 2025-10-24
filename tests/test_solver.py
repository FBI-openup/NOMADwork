from __future__ import annotations

from pathlib import Path

import pytest

from src.capacity import build_capacity_table, capacity_lookup
from src.candidates import CandidateConfig, build_candidate_index
from src.heuristics import build_warm_start
from src.models import ScheduleSegment
from src.parser import load_problem
from src.postprocess import (
    assignments_to_flow_list,
    compute_score,
    schedule_to_text,
    text_to_assignments,
)
from src.score_cli import main as score_main
from src.scorer import score_from_files
from src.solver import SolverConfig, solve_problem
from src.validation import ScheduleValidationError, validate_schedule

EXAMPLE_PATH = Path("examples/example1.in")
EXAMPLE_OUTPUT = Path("examples/example1.out")


def _load_example():
    return load_problem(EXAMPLE_PATH)


def test_parser_reads_example():
    problem = _load_example()
    assert problem.grid.rows == 3
    assert len(problem.uavs) == 9
    assert len(problem.flows) == 2


def test_capacity_pattern_matches_reference():
    problem = _load_example()
    table = build_capacity_table(problem)
    capacity_at_time0 = capacity_lookup(table, (0, 0), 0)
    capacity_at_time9 = capacity_lookup(table, (0, 0), 9)
    capacity_at_time7 = capacity_lookup(table, (0, 0), 7)
    assert pytest.approx(capacity_at_time0, rel=1e-6) == 10.0
    assert pytest.approx(capacity_at_time9, rel=1e-6) == 5.0
    assert pytest.approx(capacity_at_time7, rel=1e-6) == 0.0


def test_candidate_generation_respects_earliest_arrival():
    problem = _load_example()
    capacity_table = build_capacity_table(problem)
    candidates = build_candidate_index(problem, capacity_table, CandidateConfig(base_window=5))
    flow1 = problem.flows[0]
    slots = candidates.flows[flow1.flow_id].time_slots[(0, 0)]
    assert slots[0] == flow1.earliest_arrival((0, 0))


def test_warm_start_does_not_exceed_capacity():
    problem = _load_example()
    capacity_table = build_capacity_table(problem)
    candidates = build_candidate_index(problem, capacity_table, CandidateConfig())
    warm_start = build_warm_start(problem, candidates, capacity_table)
    usage = {}
    for segments in warm_start.assignments.values():
        for segment in segments:
            key = (segment.x, segment.y, segment.time)
            usage[key] = usage.get(key, 0.0) + segment.rate
    for (x, y, time_slot), total in usage.items():
        assert total <= capacity_lookup(capacity_table, (x, y), time_slot) + 1e-6


def test_solver_produces_valid_schedule():
    problem = _load_example()
    result = solve_problem(problem, solver_config=SolverConfig(total_time_limit=2.0))
    validate_schedule(problem, result.assignments)
    score = compute_score(problem, result.assignments)
    assert score.total >= 90.0


def test_schedule_roundtrip_parsing():
    problem = _load_example()
    result = solve_problem(problem, solver_config=SolverConfig(total_time_limit=2.0))
    flow_assignments = assignments_to_flow_list(result.assignments)
    text = schedule_to_text(flow_assignments)
    parsed = text_to_assignments(text)
    for flow_id, segments in result.assignments.items():
        rebuilt = parsed[flow_id]
        assert len(segments) == len(rebuilt)
        for original, restored in zip(segments, rebuilt):
            assert original.time == restored.time
            assert original.x == restored.x
            assert original.y == restored.y
            assert pytest.approx(original.rate, rel=1e-6, abs=1e-6) == restored.rate


def test_scoring_existing_output():
    score = score_from_files(EXAMPLE_PATH, EXAMPLE_OUTPUT, validate=True)
    assert pytest.approx(score.total, rel=1e-6) == 97.04739704739706


def test_validation_detects_illegal_multiple_cells():
    problem = _load_example()
    illegal_assignments = {
        problem.flows[0].flow_id: [
            # illegal: same time two cells
            ScheduleSegment(time=1, x=0, y=0, rate=5.0),
            ScheduleSegment(time=1, x=0, y=1, rate=5.0),
        ]
    }
    with pytest.raises(ScheduleValidationError):
        validate_schedule(problem, illegal_assignments)


def test_score_cli_reports_score(capsys):
    exit_code = score_main(
        [
            "--input",
            str(EXAMPLE_PATH),
            "--schedule",
            str(EXAMPLE_OUTPUT),
            "--validate",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Score:" in captured.out
