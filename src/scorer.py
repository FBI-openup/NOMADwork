"""Scoring and validation utilities for existing schedule files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .models import Problem, ScheduleSegment, ScoreSummary
from .parser import load_problem
from .postprocess import compute_score, text_to_assignments
from .validation import ScheduleValidationError, validate_schedule


class ScheduleFormatError(ValueError):
    """Raised when a schedule does not follow the expected text layout."""


def load_schedule(path: str | Path) -> Dict[int, List[ScheduleSegment]]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ScheduleFormatError(f"failed to read schedule from {path}: {exc}") from exc
    try:
        return text_to_assignments(text)
    except ValueError as exc:
        raise ScheduleFormatError(str(exc)) from exc


def score_schedule(
    problem: Problem,
    assignments: Dict[int, List[ScheduleSegment]],
    *,
    validate: bool = False,
) -> ScoreSummary:
    if validate:
        validate_schedule(problem, assignments)
    return compute_score(problem, assignments)


def score_from_files(
    problem_path: str | Path,
    schedule_path: str | Path,
    *,
    validate: bool = False,
) -> ScoreSummary:
    problem = load_problem(problem_path)
    assignments = load_schedule(schedule_path)
    return score_schedule(problem, assignments, validate=validate)
