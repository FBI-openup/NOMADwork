"""Schedule utilities: merging, scoring, and formatting."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .models import Flow, FlowAssignment, Problem, ScheduleSegment, ScoreSlice, ScoreSummary

EPSILON = 1e-6
T_MAX = 10.0
DELAY_WEIGHT = 0.2
DISTANCE_WEIGHT = 0.3
LANDING_WEIGHT = 0.1
THROUGHPUT_WEIGHT = 0.4
SCORE_SCALE = 100.0
DISTANCE_ALPHA = 0.1


def merge_assignments(
    destination: Dict[int, List[ScheduleSegment]],
    source: Dict[int, List[ScheduleSegment]],
) -> None:
    for flow_id, segments in source.items():
        destination.setdefault(flow_id, []).extend(segments)


def normalize_assignments(assignments: Dict[int, List[ScheduleSegment]]) -> None:
    for segments in assignments.values():
        segments.sort(key=lambda seg: (seg.time, seg.x, seg.y))


def _group_by_time(segments: Iterable[ScheduleSegment]) -> Dict[int, List[ScheduleSegment]]:
    grouped: Dict[int, List[ScheduleSegment]] = defaultdict(list)
    for segment in segments:
        grouped[segment.time].append(segment)
    return grouped


def _score_flow(flow: Flow, segments: List[ScheduleSegment]) -> Tuple[float, float, float, float, float]:
    grouped = _group_by_time(segments)
    total_transmitted = sum(segment.rate for segment in segments)
    throughput_score = min(1.0, total_transmitted / flow.volume) if flow.volume > 0 else 0.0

    delay_score = 0.0
    distance_score = 0.0
    landing_sequence: List[Tuple[int, int]] = []
    previous_cell: Tuple[int, int] | None = None

    for time_slot in sorted(grouped):
        bucket = grouped[time_slot]
        bucket_total = sum(segment.rate for segment in bucket)
        delay_score += (T_MAX / (time_slot + T_MAX)) * (bucket_total / flow.volume)

        for segment in bucket:
            manhattan = abs(flow.access.x - segment.x) + abs(flow.access.y - segment.y)
            distance_score += (segment.rate / flow.volume) * (2.0 ** (-DISTANCE_ALPHA * manhattan))

        representative = max(bucket, key=lambda item: item.rate)
        cell = (representative.x, representative.y)
        if previous_cell is None or cell != previous_cell:
            landing_sequence.append(cell)
            previous_cell = cell

    landing_score = 1.0 / max(1, len(landing_sequence))

    composite = SCORE_SCALE * (
        THROUGHPUT_WEIGHT * throughput_score
        + DELAY_WEIGHT * delay_score
        + DISTANCE_WEIGHT * distance_score
        + LANDING_WEIGHT * landing_score
    )
    return composite, throughput_score, delay_score, distance_score, landing_score


def compute_score(problem: Problem, assignments: Dict[int, List[ScheduleSegment]]) -> ScoreSummary:
    total_volume = sum(flow.volume for flow in problem.flows)
    if total_volume <= EPSILON:
        return ScoreSummary(total=0.0, by_flow=[])

    slices: List[ScoreSlice] = []
    total_score = 0.0

    for flow in problem.flows:
        segments = assignments.get(flow.flow_id, [])
        if not segments:
            slices.append(
                ScoreSlice(
                    flow_id=flow.flow_id,
                    traffic=0.0,
                    delay=0.0,
                    distance=0.0,
                    landing=0.0,
                    weight=flow.volume / total_volume,
                )
            )
            continue

        composite, throughput, delay, distance, landing = _score_flow(flow, segments)
        weight = flow.volume / total_volume
        total_score += weight * composite
        slices.append(
            ScoreSlice(
                flow_id=flow.flow_id,
                traffic=throughput,
                delay=delay,
                distance=distance,
                landing=landing,
                weight=weight,
            )
        )

    return ScoreSummary(total=total_score, by_flow=slices)


def assignments_to_flow_list(assignments: Dict[int, List[ScheduleSegment]]) -> List[FlowAssignment]:
    flow_assignments: List[FlowAssignment] = []
    for flow_id in sorted(assignments):
        segments = sorted(assignments[flow_id], key=lambda seg: (seg.time, seg.x, seg.y))
        flow_assignments.append(FlowAssignment(flow_id=flow_id, segments=segments))
    return flow_assignments


def text_to_assignments(text: str) -> Dict[int, List[ScheduleSegment]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    index = 0
    assignments: Dict[int, List[ScheduleSegment]] = {}

    while index < len(lines):
        header = lines[index].split()
        index += 1
        if len(header) != 2:
            raise ValueError(f"invalid schedule header line: {lines[index - 1]!r}")
        flow_id = int(header[0])
        segment_count = int(header[1])
        segments: List[ScheduleSegment] = []
        for _ in range(segment_count):
            if index >= len(lines):
                raise ValueError("unexpected end of schedule while reading segments")
            parts = lines[index].split()
            index += 1
            if len(parts) != 4:
                raise ValueError(f"invalid segment line: {lines[index - 1]!r}")
            time_slot = int(parts[0])
            x_coord = int(parts[1])
            y_coord = int(parts[2])
            rate = float(parts[3])
            segments.append(ScheduleSegment(time=time_slot, x=x_coord, y=y_coord, rate=rate))
        assignments[flow_id] = segments

    return assignments


def schedule_to_text(flow_assignments: List[FlowAssignment]) -> str:
    lines: List[str] = []
    for assignment in sorted(flow_assignments, key=lambda fa: fa.flow_id):
        segments = sorted(assignment.segments, key=lambda seg: (seg.time, seg.x, seg.y))
        lines.append(f"{assignment.flow_id} {len(segments)}")
        for segment in segments:
            lines.append(f"{segment.time} {segment.x} {segment.y} {segment.rate:.6f}")
    return "\n".join(lines)


def schedule_to_dict(flow_assignments: List[FlowAssignment]) -> dict:
    return {
        "assignments": [
            {
                "flow_id": assignment.flow_id,
                "segments": [
                    {
                        "time": segment.time,
                        "x": segment.x,
                        "y": segment.y,
                        "rate": segment.rate,
                    }
                    for segment in assignment.segments
                ],
            }
            for assignment in flow_assignments
        ]
    }
