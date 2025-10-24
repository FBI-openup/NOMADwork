"""Input parsing for the Huawei traffic allocation challenge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

from .models import AccessPoint, Flow, Grid, LandingZone, Problem, UAV


class ProblemFormatError(ValueError):
    """Raised when an input file cannot be parsed."""


def _tokens_from_text(raw_text: str) -> List[str]:
    tokens = raw_text.split()
    if not tokens:
        raise ProblemFormatError("input is empty; expected problem definition")
    return tokens


def _take(tokens: Sequence[str], index: int) -> str:
    try:
        return tokens[index]
    except IndexError as exc:
        raise ProblemFormatError("unexpected end of input while parsing problem definition") from exc


def _parse_from_tokens(tokens: Sequence[str]) -> Problem:
    pointer = 0

    def next_token() -> str:
        nonlocal pointer
        token = _take(tokens, pointer)
        pointer += 1
        return token

    rows = int(next_token())
    cols = int(next_token())
    flow_count = int(next_token())
    horizon = int(next_token())

    cell_total = rows * cols
    uavs: List[UAV] = []
    for _ in range(cell_total):
        x = int(next_token())
        y = int(next_token())
        peak_bandwidth = float(next_token())
        phase = int(next_token())
        uavs.append(UAV(x=x, y=y, peak_bandwidth=peak_bandwidth, phase=phase))

    flows: List[Flow] = []
    for _ in range(flow_count):
        flow_id = int(next_token())
        access_x = int(next_token())
        access_y = int(next_token())
        start_time = int(next_token())
        volume = float(next_token())
        zone_x_min = int(next_token())
        zone_y_min = int(next_token())
        zone_x_max = int(next_token())
        zone_y_max = int(next_token())
        access_point = AccessPoint(x=access_x, y=access_y)
        landing_zone = LandingZone(
            x_min=zone_x_min,
            y_min=zone_y_min,
            x_max=zone_x_max,
            y_max=zone_y_max,
        )
        flows.append(
            Flow(
                flow_id=flow_id,
                access=access_point,
                start_time=start_time,
                volume=volume,
                zone=landing_zone,
            )
        )

    grid = Grid(rows=rows, cols=cols, horizon=horizon)
    return Problem(grid=grid, uavs=uavs, flows=flows)


def parse_problem_from_text(raw_text: str) -> Problem:
    """Parse either whitespace-delimited `.in` or JSON definitions."""
    stripped = raw_text.lstrip()
    if not stripped:
        raise ProblemFormatError("input is empty; expected problem definition")
    if stripped[0] in "{[":
        payload = json.loads(raw_text)
        return _parse_from_json(payload)
    tokens = _tokens_from_text(raw_text)
    return _parse_from_tokens(tokens)


def _parse_from_json(payload: dict) -> Problem:
    try:
        grid_payload = payload["grid"]
        uav_payloads = payload["uavs"]
        flow_payloads = payload["flows"]
    except KeyError as exc:
        raise ProblemFormatError(f"missing required field: {exc}") from exc

    grid = Grid(
        rows=int(grid_payload["rows"]),
        cols=int(grid_payload["cols"]),
        horizon=int(grid_payload["horizon"]),
    )

    uavs = [
        UAV(
            x=int(item["x"]),
            y=int(item["y"]),
            peak_bandwidth=float(item["peak_bandwidth"]),
            phase=int(item["phase"]),
        )
        for item in uav_payloads
    ]

    flows = [
        Flow(
            flow_id=int(item["id"]),
            access=AccessPoint(
                x=int(item["access"]["x"]),
                y=int(item["access"]["y"]),
            ),
            start_time=int(item["start_time"]),
            volume=float(item["volume"]),
            zone=LandingZone(
                x_min=int(item["zone"]["x_min"]),
                y_min=int(item["zone"]["y_min"]),
                x_max=int(item["zone"]["x_max"]),
                y_max=int(item["zone"]["y_max"]),
            ),
        )
        for item in flow_payloads
    ]

    return Problem(grid=grid, uavs=uavs, flows=flows)


def load_problem(path: str | Path) -> Problem:
    """Load a problem definition from `path` or raise `ProblemFormatError`."""
    raw_text = Path(path).read_text(encoding="utf-8")
    return parse_problem_from_text(raw_text)
