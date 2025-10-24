#!/usr/bin/env python3
"""
Single-file submission script for the Huawei UAV traffic allocation challenge.

Reads the instance from stdin (either .in plaintext or JSON) and writes the
schedule to stdout with no auxiliary logging.
Usage:
    python submission.py < input.in > output.out

"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

try:
    import pulp
except ModuleNotFoundError:  # pragma: no cover - handled via warm-start fallback
    pulp = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


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
    access_x: int
    access_y: int
    start_time: int
    volume: float
    zone: LandingZone

    def travel_time(self, cell: Tuple[int, int]) -> int:
        return abs(self.access_x - cell[0]) + abs(self.access_y - cell[1])

    def earliest_arrival(self, cell: Tuple[int, int]) -> int:
        return self.start_time + self.travel_time(cell)


@dataclass(slots=True)
class Problem:
    grid: Grid
    uavs: List[UAV]
    flows: List[Flow]


@dataclass(slots=True)
class ScheduleSegment:
    time: int
    x: int
    y: int
    rate: float


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------


def _parse_tokens(tokens: List[str]) -> Problem:
    idx = 0

    def next_token() -> str:
        nonlocal idx
        if idx >= len(tokens):
            raise ValueError("unexpected end of input while parsing instance")
        token = tokens[idx]
        idx += 1
        return token

    rows = int(next_token())
    cols = int(next_token())
    flow_count = int(next_token())
    horizon = int(next_token())

    uavs: List[UAV] = []
    for _ in range(rows * cols):
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
        x_min = int(next_token())
        y_min = int(next_token())
        x_max = int(next_token())
        y_max = int(next_token())
        flows.append(
            Flow(
                flow_id=flow_id,
                access_x=access_x,
                access_y=access_y,
                start_time=start_time,
                volume=volume,
                zone=LandingZone(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            )
        )

    grid = Grid(rows=rows, cols=cols, horizon=horizon)
    return Problem(grid=grid, uavs=uavs, flows=flows)


def _parse_json(payload: dict) -> Problem:
    grid = payload["grid"]
    uavs = [
        UAV(
            x=int(item["x"]),
            y=int(item["y"]),
            peak_bandwidth=float(item["peak_bandwidth"]),
            phase=int(item["phase"]),
        )
        for item in payload["uavs"]
    ]
    flows = [
        Flow(
            flow_id=int(item["id"]),
            access_x=int(item["access"]["x"]),
            access_y=int(item["access"]["y"]),
            start_time=int(item["start_time"]),
            volume=float(item["volume"]),
            zone=LandingZone(
                x_min=int(item["zone"]["x_min"]),
                y_min=int(item["zone"]["y_min"]),
                x_max=int(item["zone"]["x_max"]),
                y_max=int(item["zone"]["y_max"]),
            ),
        )
        for item in payload["flows"]
    ]
    return Problem(
        grid=Grid(rows=int(grid["rows"]), cols=int(grid["cols"]), horizon=int(grid["horizon"])),
        uavs=uavs,
        flows=flows,
    )


def parse_problem(raw_text: str) -> Problem:
    stripped = raw_text.lstrip()
    if not stripped:
        raise ValueError("input is empty; expected problem definition")
    if stripped[0] in "{[":
        return _parse_json(json.loads(raw_text))
    return _parse_tokens(raw_text.split())


# ---------------------------------------------------------------------------
# Capacity timeline
# ---------------------------------------------------------------------------


CAPACITY_PATTERN = (0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0)
EPSILON = 1e-6

# Candidate pruning controls
BASE_WINDOW = 30
EXPAND_STEP = 10
MAX_WINDOW = 80
CAPACITY_BUFFER = 1.05
MAX_SLOTS_PER_CELL = 40
MAX_CELLS_PER_FLOW = 12


def capacity_at(uav: UAV, time_slot: int) -> float:
    index = (uav.phase + time_slot) % len(CAPACITY_PATTERN)
    return uav.peak_bandwidth * CAPACITY_PATTERN[index]


def build_capacity_table(problem: Problem) -> Dict[Tuple[int, int], List[float]]:
    horizon = problem.grid.horizon
    table: Dict[Tuple[int, int], List[float]] = {}
    for uav in problem.uavs:
        table[(uav.x, uav.y)] = [capacity_at(uav, t) for t in range(horizon)]
    return table


def capacity_lookup(table: Dict[Tuple[int, int], List[float]], cell: Tuple[int, int], time_slot: int) -> float:
    return table[cell][time_slot]


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FlowCandidate:
    flow: Flow
    time_slots: Dict[Tuple[int, int], List[int]]


@dataclass(slots=True)
class CandidateIndex:
    flows: Dict[int, FlowCandidate]


def _collect_time_slots(
    flow: Flow,
    cell: Tuple[int, int],
    horizon: int,
    table: Dict[Tuple[int, int], List[float]],
) -> List[int]:
    earliest = flow.earliest_arrival(cell)
    if earliest >= horizon:
        return []

    window = BASE_WINDOW
    total_capacity = 0.0
    selected: List[int] = []
    seen = set()

    def consider(limit: int) -> float:
        nonlocal total_capacity
        for t in range(earliest, min(limit, horizon - 1) + 1):
            if t in seen:
                continue
            seen.add(t)
            cap = table[cell][t]
            if cap > 1e-6:
                selected.append(t)
                total_capacity += cap
        return total_capacity

    consider(earliest + window)
    while (
        total_capacity + 1e-6 < flow.volume * CAPACITY_BUFFER
        and window < MAX_WINDOW
        and earliest + window < horizon
    ):
        window = min(MAX_WINDOW, window + EXPAND_STEP)
        consider(earliest + window)

    selected.sort()
    if len(selected) > MAX_SLOTS_PER_CELL:
        selected = selected[:MAX_SLOTS_PER_CELL]
    return selected


def build_candidate_index(problem: Problem, table: Dict[Tuple[int, int], List[float]]) -> CandidateIndex:
    horizon = problem.grid.horizon
    flow_map: Dict[int, FlowCandidate] = {}
    for flow in problem.flows:
        slot_map: Dict[Tuple[int, int], List[int]] = {}
        for cell in flow.zone.cells():
            if cell not in table:
                continue
            slots = _collect_time_slots(flow, cell, horizon, table)
            if slots:
                slot_map[cell] = slots
        if len(slot_map) > MAX_CELLS_PER_FLOW:
            ranked = sorted(
                slot_map.items(),
                key=lambda item: _score_cell(flow, item[1], item[0], table),
                reverse=True,
            )
            slot_map = dict(ranked[:MAX_CELLS_PER_FLOW])
        flow_map[flow.flow_id] = FlowCandidate(flow=flow, time_slots=slot_map)
    return CandidateIndex(flows=flow_map)


# ---------------------------------------------------------------------------
# Clustering (group flows with overlapping landing zones)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Cluster:
    flow_ids: List[int]
    landing_cells: List[Tuple[int, int]]


def cluster_flows(index: CandidateIndex) -> List[Cluster]:
    cell_to_flows: Dict[Tuple[int, int], List[int]] = {}
    for flow_id, fc in index.flows.items():
        for cell in fc.time_slots:
            cell_to_flows.setdefault(cell, []).append(flow_id)

    visited = set()
    clusters: List[Cluster] = []

    for flow_id in index.flows:
        if flow_id in visited:
            continue
        queue = [flow_id]
        pointer = 0
        flow_ids: List[int] = []
        landing_cells = set()

        while pointer < len(queue):
            current = queue[pointer]
            pointer += 1
            if current in visited:
                continue
            visited.add(current)
            flow_ids.append(current)
            fc = index.flows[current]
            for cell in fc.time_slots:
                landing_cells.add(cell)
                for neighbour in cell_to_flows.get(cell, []):
                    if neighbour not in visited:
                        queue.append(neighbour)

        clusters.append(
            Cluster(
                flow_ids=sorted(flow_ids),
                landing_cells=sorted(landing_cells),
            )
        )

    clusters.sort(key=lambda c: len(c.flow_ids), reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# Warm-start heuristic (greedy but respects constraints)
# ---------------------------------------------------------------------------


def _score_cell(flow: Flow, slots: List[int], cell: Tuple[int, int], table) -> float:
    total = sum(table[cell][t] for t in slots)
    return total / max(1, flow.travel_time(cell))


def build_warm_start(problem: Problem, index: CandidateIndex, table: Dict[Tuple[int, int], List[float]]) -> Dict[int, List[ScheduleSegment]]:
    remaining = {cell: caps.copy() for cell, caps in table.items()}
    assignments: Dict[int, List[ScheduleSegment]] = {}

    for flow in sorted(problem.flows, key=lambda f: f.start_time):
        fc = index.flows.get(flow.flow_id)
        if fc is None:
            assignments[flow.flow_id] = []
            continue

        ordered_cells = sorted(
            fc.time_slots.items(),
            key=lambda item: _score_cell(flow, item[1], item[0], table),
            reverse=True,
        )

        segments: List[ScheduleSegment] = []
        remaining_volume = flow.volume
        used_times = set()

        for cell, slots in ordered_cells:
            if remaining_volume <= EPSILON:
                break
            for time_slot in slots:
                if time_slot in used_times:
                    continue
                available = remaining[cell][time_slot]
                if available <= EPSILON:
                    continue
                take = min(available, remaining_volume)
                if take <= EPSILON:
                    continue
                remaining[cell][time_slot] -= take
                segments.append(ScheduleSegment(time=time_slot, x=cell[0], y=cell[1], rate=take))
                used_times.add(time_slot)
                remaining_volume -= take
                if remaining_volume <= EPSILON:
                    break

        segments.sort(key=lambda seg: (seg.time, seg.x, seg.y))
        assignments[flow.flow_id] = segments

    return assignments


# ---------------------------------------------------------------------------
# MILP solver (cluster-by-cluster with fallback to warm start)
# ---------------------------------------------------------------------------


def _build_indices(cluster: Cluster, index: CandidateIndex):
    x_keys: List[Tuple[int, Tuple[int, int], int]] = []
    flow_time: Dict[Tuple[int, int], List[Tuple[int, Tuple[int, int], int]]] = {}
    cell_time: Dict[Tuple[Tuple[int, int], int], List[Tuple[int, Tuple[int, int], int]]] = {}
    flow_cell: Dict[Tuple[int, Tuple[int, int]], List[Tuple[int, Tuple[int, int], int]]] = {}
    y_keys: List[Tuple[int, Tuple[int, int]]] = []
    seen_y = set()

    for flow_id in cluster.flow_ids:
        fc = index.flows.get(flow_id)
        if fc is None:
            continue
        for cell, slots in fc.time_slots.items():
            for time_slot in slots:
                key = (flow_id, cell, time_slot)
                x_keys.append(key)
                flow_time.setdefault((flow_id, time_slot), []).append(key)
                cell_time.setdefault((cell, time_slot), []).append(key)
                flow_cell.setdefault((flow_id, cell), []).append(key)
                if (flow_id, cell) not in seen_y:
                    seen_y.add((flow_id, cell))
                    y_keys.append((flow_id, cell))
    return x_keys, flow_time, cell_time, flow_cell, y_keys


def _x_reward(flow: Flow, cell: Tuple[int, int], time_slot: int) -> float:
    throughput = 100.0 * 0.4 / max(flow.volume, 1e-6)
    delay = 100.0 * 0.2 * (10.0 / (time_slot + 10.0)) / max(flow.volume, 1e-6)
    distance = 100.0 * 0.3 * (2.0 ** (-0.1 * flow.travel_time(cell))) / max(flow.volume, 1e-6)
    return throughput + delay + distance


def _landing_penalty(fc: FlowCandidate) -> float:
    return 100.0 * 0.1 / max(1, len(fc.time_slots))


def _fallback(cluster: Cluster, warm_start: Dict[int, List[ScheduleSegment]]) -> Dict[int, List[ScheduleSegment]]:
    result: Dict[int, List[ScheduleSegment]] = {}
    for flow_id in cluster.flow_ids:
        segments = warm_start.get(flow_id, [])
        result[flow_id] = [ScheduleSegment(seg.time, seg.x, seg.y, seg.rate) for seg in segments]
    return result


def solve_cluster(
    cluster: Cluster,
    problem: Problem,
    index: CandidateIndex,
    table: Dict[Tuple[int, int], List[float]],
    warm_start: Dict[int, List[ScheduleSegment]],
) -> Dict[int, List[ScheduleSegment]]:
    if pulp is None:
        return _fallback(cluster, warm_start)

    flow_lookup = {flow.flow_id: flow for flow in problem.flows if flow.flow_id in cluster.flow_ids}
    x_keys, flow_time, cell_time, flow_cell, y_keys = _build_indices(cluster, index)

    model = pulp.LpProblem("cluster", pulp.LpMaximize)
    x_vars = pulp.LpVariable.dicts("x", x_keys, lowBound=0.0)
    u_vars = pulp.LpVariable.dicts("u", x_keys, lowBound=0.0, upBound=1.0, cat="Binary")
    y_vars = pulp.LpVariable.dicts("y", y_keys, lowBound=0.0, upBound=1.0, cat="Binary")
    drop_vars = {fid: pulp.LpVariable(f"drop_{fid}", lowBound=0.0) for fid in cluster.flow_ids}

    for key in x_keys:
        flow_id, cell, time_slot = key
        capacity = capacity_lookup(table, cell, time_slot)
        model += x_vars[key] <= capacity * u_vars[key]

    for (flow_id, time_slot), keys in flow_time.items():
        model += pulp.lpSum(u_vars[key] for key in keys) <= 1

    for flow_id in cluster.flow_ids:
        flow = flow_lookup[flow_id]
        x_terms = [x_vars[key] for key in x_keys if key[0] == flow_id]
        model += pulp.lpSum(x_terms) + drop_vars[flow_id] == flow.volume

    for (flow_id, cell), keys in flow_cell.items():
        if not keys:
            continue
        model += y_vars[(flow_id, cell)] <= pulp.lpSum(u_vars[key] for key in keys)
        for key in keys:
            model += u_vars[key] <= y_vars[(flow_id, cell)]

    for (cell, time_slot), keys in cell_time.items():
        capacity = capacity_lookup(table, cell, time_slot)
        model += pulp.lpSum(x_vars[key] for key in keys) <= capacity

    objective_terms = []
    for key in x_keys:
        flow_id, cell, time_slot = key
        flow = flow_lookup[flow_id]
        reward = _x_reward(flow, cell, time_slot)
        objective_terms.append(reward * x_vars[key])

    landing_penalties = []
    for flow_id, cell in y_keys:
        fc = index.flows[flow_id]
        landing_penalties.append(_landing_penalty(fc) * y_vars[(flow_id, cell)])

    slack_penalties = [1e5 * drop_vars[fid] for fid in cluster.flow_ids]

    model += pulp.lpSum(objective_terms) - pulp.lpSum(landing_penalties) - pulp.lpSum(slack_penalties)

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=2, gapRel=0.05)
    status = model.solve(solver)
    if pulp.LpStatus[status] not in {"Optimal", "Feasible"}:
        return _fallback(cluster, warm_start)

    assignments: Dict[int, List[ScheduleSegment]] = {fid: [] for fid in cluster.flow_ids}
    for key in x_keys:
        value = x_vars[key].value() or 0.0
        if value <= EPSILON:
            continue
        fid, cell, time_slot = key
        assignments.setdefault(fid, []).append(
            ScheduleSegment(time=time_slot, x=cell[0], y=cell[1], rate=value)
        )

    for segments in assignments.values():
        segments.sort(key=lambda seg: (seg.time, seg.x, seg.y))

    return assignments


# ---------------------------------------------------------------------------
# Problem solve orchestration
# ---------------------------------------------------------------------------


def solve_problem(problem: Problem) -> Dict[int, List[ScheduleSegment]]:
    capacity_table = build_capacity_table(problem)
    candidate_index = build_candidate_index(problem, capacity_table)
    clusters = cluster_flows(candidate_index)
    warm_start = build_warm_start(problem, candidate_index, capacity_table)

    assignments: Dict[int, List[ScheduleSegment]] = {}
    for cluster in clusters:
        solution = solve_cluster(cluster, problem, candidate_index, capacity_table, warm_start)
        for flow_id, segments in solution.items():
            assignments.setdefault(flow_id, []).extend(segments)

    for flow in problem.flows:
        segs = assignments.get(flow.flow_id, [])
        segs.sort(key=lambda seg: (seg.time, seg.x, seg.y))
        assignments[flow.flow_id] = segs

    return assignments


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def schedule_to_text(problem: Problem, assignments: Dict[int, List[ScheduleSegment]]) -> str:
    lines: List[str] = []
    for flow in problem.flows:
        segments = assignments.get(flow.flow_id, [])
        lines.append(f"{flow.flow_id} {len(segments)}")
        for segment in segments:
            lines.append(f"{segment.time} {segment.x} {segment.y} {segment.rate:.6f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    raw = sys.stdin.read()
    if not raw.strip():
        return
    problem = parse_problem(raw)
    assignments = solve_problem(problem)
    text = schedule_to_text(problem, assignments)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
