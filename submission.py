#!/usr/bin/env python3
"""
Single-file submission script for the Huawei UAV traffic allocation challenge.

Reads the instance from stdin (either .in plaintext or JSON) and writes the
schedule to stdout with no auxiliary logging.
Usage:
    python submission.py < input.in > output.out

python submission.py < examples/gen1.in > examples/gen1V3.out

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

# Candidate generation / solver tuning
INITIAL_WINDOW = 40              # seconds inspected before expansion
WINDOW_EXPAND_STEP = 20          # growth step when capacity is insufficient
CAPACITY_BUFFER = 2.0            # expand more aggressively to include more slots
MAX_WINDOW_SECONDS = None        # optional hard ceiling (None -> horizon)
MAX_SLOTS_PER_CELL = 200         # keep more time slots per landing cell
MAX_CELLS_PER_FLOW = 50          # cap on landing cells retained per flow

# Objective tuning
DELAY_WEIGHT_MULTIPLIER = 1    # emphasise early transmissions
LANDING_PENALTY_MULTIPLIER = 1 # stronger penalty for revisiting multiple landings
LANDING_EXTRA_PENALTY = 100.0 * 0.1 * LANDING_PENALTY_MULTIPLIER


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


# Candidate prioritisation helper
def _cell_priority_score(flow: Flow, slots: List[int], cell: Tuple[int, int], table: Dict[Tuple[int, int], List[float]]) -> Tuple[float, int]:
    # Combine early availability, near-term capacity, and distance preference
    earliest = slots[0]
    preview = slots[: min(20, len(slots))]
    total_capacity = sum(table[cell][t] for t in preview)
    h = flow.travel_time(cell)
    near = 2.0 ** (-0.1 * float(h))
    # Normalized capacity score
    cap_norm = total_capacity / (total_capacity + 1.0)
    composite = 0.5 * (1.0 / (1 + max(0, earliest - flow.start_time))) + 0.3 * cap_norm + 0.2 * near
    # Sort by composite desc; tie-break by earlier time
    return (composite, -earliest)


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

    window = INITIAL_WINDOW
    total_capacity = 0.0
    selected: List[int] = []
    seen = set()

    max_window_limit = horizon - 1 if MAX_WINDOW_SECONDS is None else min(horizon - 1, earliest + MAX_WINDOW_SECONDS)

    def consider(limit: int) -> float:
        nonlocal total_capacity
        for t in range(earliest, min(limit, max_window_limit) + 1):
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
        and earliest + window < max_window_limit
    ):
        window = min(max_window_limit - earliest, window + WINDOW_EXPAND_STEP)
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
                key=lambda item: _cell_priority_score(flow, item[1], item[0], table),
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


# -----------------------------
# Greedy warm start (replica of prior_solver)
# -----------------------------

# Priority weights (aligned with scoring weights, scaled)
W_TRAFFIC = 40.0
W_DELAY = 20.0
W_DIST = 30.0
W_LAND = 10.0

# Dynamic bonuses
INERTIA_BONUS = 4.0
ETA_BONUS_B = 3.0
ETA_BONUS_HB = 1.0
COMPLETE_BONUS_ALPHA = 0.05

# Distance decay and candidate pruning
BETA_HOP = 0.1
GREEDY_TOP_K = 99

# Short-horizon capacity lookahead bonus (encourage near-term capacity)
LOOKAHEAD_WINDOW = 7  # seconds ahead including current
LOOKAHEAD_BONUS_SCALE = 0.5


def _delay_weight(t: int) -> float:
    return 10.0 / (t + 10.0)


def _distance_score(hops: int) -> float:
    return 2.0 ** (-BETA_HOP * float(hops))


def _landing_switch_penalty(k_current: int) -> float:
    k = max(1, k_current)
    return (1.0 / (k + 1)) - (1.0 / k)


def _build_greedy_warm_start(problem: Problem, table: Dict[Tuple[int, int], List[float]]) -> Dict[int, List[ScheduleSegment]]:
    horizon = problem.grid.horizon
    # Map cell -> UAV to access peak bandwidth
    cell_to_uav: Dict[Tuple[int, int], UAV] = {(u.x, u.y): u for u in problem.uavs}

    # Build candidate landing cells per flow using hybrid heuristic
    flow_candidates: Dict[int, List[Tuple[int, int]]] = {}
    hop_min: Dict[Tuple[int, Tuple[int, int]], int] = {}
    eta_map: Dict[Tuple[int, Tuple[int, int]], int] = {}

    for flow in problem.flows:
        src = (flow.access_x, flow.access_y)
        candidates: List[Tuple[float, Tuple[int, int], int]] = []
        for x in range(flow.zone.x_min, flow.zone.x_max + 1):
            for y in range(flow.zone.y_min, flow.zone.y_max + 1):
                cell = (x, y)
                uav = cell_to_uav.get(cell)
                if uav is None:
                    continue
                h = abs(src[0] - x) + abs(src[1] - y)
                capacity_signal = uav.peak_bandwidth
                heuristic = 0.7 * _distance_score(h) + 0.3 * (capacity_signal / (capacity_signal + 1.0))
                candidates.append((heuristic, cell, h))
        candidates.sort(key=lambda it: (-it[0], it[2]))
        top_cells = [cell for _, cell, _ in candidates[: max(1, GREEDY_TOP_K)]]
        flow_candidates[flow.flow_id] = top_cells
        for cell in top_cells:
            h = abs(src[0] - cell[0]) + abs(src[1] - cell[1])
            hop_min[(flow.flow_id, cell)] = h
            eta_map[(flow.flow_id, cell)] = flow.start_time + h

    # Previous second landing per flow for inertia bonus
    last_used_map: Dict[int, Tuple[int, int]] = {}

    def capacity_window_flag(cell: Tuple[int, int], t: int) -> int:
        uav = cell_to_uav[cell]
        # Next second capacity and normalized against peak
        t1 = min(horizon - 1, t + 1)
        cnext = table[cell][t1]
        if cnext <= 1e-9:
            return 0
        if abs(cnext - uav.peak_bandwidth) < 1e-9:
            return 2
        # Check half-bandwidth window (within tolerance)
        if abs(cnext - 0.5 * uav.peak_bandwidth) < 1e-9:
            return 1
        # Treat any nonzero as half for purposes of foresight
        return 1

    # Initialize per-flow state
    class _FState:
        __slots__ = ("remain", "k", "last_landing", "active", "done", "records")

        def __init__(self, volume: float):
            self.remain = float(volume)
            self.k = 0
            self.last_landing: Tuple[int, int] | None = None
            self.active = False
            self.done = False
            self.records: List[ScheduleSegment] = []

    state: Dict[int, _FState] = {flow.flow_id: _FState(flow.volume) for flow in problem.flows}

    # Per-time capacities
    remaining_capacity: Dict[Tuple[int, int], List[float]] = {cell: caps.copy() for cell, caps in table.items()}

    for t in range(horizon):
        # Activate flows that have started
        for flow in problem.flows:
            st = state[flow.flow_id]
            if not st.active and (t >= flow.start_time) and not st.done:
                st.active = True

        # Build options list across all flows and candidate cells
        options: List[Tuple[Tuple[float, float, float, float, float], int, Tuple[int, int]]] = []

        for flow in problem.flows:
            st = state[flow.flow_id]
            if not st.active or st.done or st.remain <= 1e-9:
                continue

            for cell in flow_candidates.get(flow.flow_id, []):
                if t < eta_map[(flow.flow_id, cell)]:
                    continue
                cap = remaining_capacity[cell][t]
                if cap <= 1e-9:
                    continue

                # Compute priority
                base_traffic = W_TRAFFIC * (1.0 / max(flow.volume, 1e-9))
                base_delay = W_DELAY * _delay_weight(t) * (1.0 / max(flow.volume, 1e-9))
                h = hop_min[(flow.flow_id, cell)]
                base_dist = W_DIST * _distance_score(h)
                land_term = 0.0
                if st.last_landing is not None and st.last_landing != cell:
                    land_term = W_LAND * _landing_switch_penalty(st.k)
                bonus = 0.0
                if last_used_map.get(flow.flow_id) == cell:
                    bonus += INERTIA_BONUS
                flag = capacity_window_flag(cell, t)
                if flag == 2:
                    bonus += ETA_BONUS_B
                elif flag == 1:
                    bonus += ETA_BONUS_HB
                remain_ratio = st.remain / max(flow.volume, 1e-9)
                bonus += COMPLETE_BONUS_ALPHA * (1.0 - remain_ratio)

                # Lookahead capacity sum bonus (normalized by peak)
                uav = cell_to_uav[cell]
                wsum = 0.0
                for dt in range(0, LOOKAHEAD_WINDOW + 1):
                    tt = t + dt
                    if tt >= horizon:
                        break
                    wsum += table[cell][tt]
                if uav.peak_bandwidth > 1e-9:
                    bonus += LOOKAHEAD_BONUS_SCALE * (wsum / (uav.peak_bandwidth * max(1, LOOKAHEAD_WINDOW + 1)))

                priority = base_traffic + base_delay + base_dist + land_term + bonus
                # Tie-break key
                eta_val = eta_map[(flow.flow_id, cell)]
                dist_score = _distance_score(h)
                inv_rem = -1.0 / st.remain if st.remain > 0 else 0.0
                key = (priority, -float(eta_val), -dist_score, inv_rem, -float(flow.flow_id))
                options.append((key, flow.flow_id, cell))

        # Sort options by priority key desc
        options.sort(key=lambda it: it[0], reverse=True)

        assigned_flows: set[int] = set()
        last_used_map.clear()

        for _, fid, cell in options:
            if fid in assigned_flows:
                continue
            st = state[fid]
            if st.remain <= 1e-9:
                continue
            cap = remaining_capacity[cell][t]
            if cap <= 1e-9:
                continue
            z = min(st.remain, cap)
            if z <= 1e-9:
                continue
            remaining_capacity[cell][t] -= z
            st.remain -= z
            st.records.append(ScheduleSegment(time=t, x=cell[0], y=cell[1], rate=z))
            assigned_flows.add(fid)
            last_used_map[fid] = cell
            if st.last_landing is None:
                st.last_landing = cell
                st.k = 1
            elif st.last_landing != cell:
                st.k += 1
                st.last_landing = cell
            if st.remain <= 1e-9:
                st.done = True

    assignments: Dict[int, List[ScheduleSegment]] = {}
    for flow in problem.flows:
        recs = state[flow.flow_id].records
        recs.sort(key=lambda seg: (seg.time, seg.x, seg.y))
        assignments[flow.flow_id] = recs
    return assignments


def build_warm_start(
    problem: Problem,
    index: CandidateIndex,
    table: Dict[Tuple[int, int], List[float]],
) -> Dict[int, List[ScheduleSegment]]:
    # Use the prior_solver greedy as warm start
    return _build_greedy_warm_start(problem, table)


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
    ROLLING_WINDOW = 200

    for flow_id in cluster.flow_ids:
        fc = index.flows.get(flow_id)
        if fc is None:
            continue
        for cell, slots in fc.time_slots.items():
            # Limit to a rolling window since earliest arrival to reduce problem size
            from_flow = fc.flow
            ea = from_flow.earliest_arrival(cell)
            limit = ea + ROLLING_WINDOW
            for time_slot in slots:
                if time_slot > limit:
                    continue
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
    delay = (100.0 * 0.2 * (10.0 / (time_slot + 10.0)) / max(flow.volume, 1e-6)) * DELAY_WEIGHT_MULTIPLIER
    distance = 100.0 * 0.3 * (2.0 ** (-0.1 * flow.travel_time(cell))) / max(flow.volume, 1e-6)
    return throughput + delay + distance


def _landing_penalty(fc: FlowCandidate) -> float:
    return 0.0


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

    # Link landing count to piecewise 1/k score via z-variables
    flow_to_y = {fid: [] for fid in cluster.flow_ids}
    for fid, cell in y_keys:
        flow_to_y.setdefault(fid, []).append((fid, cell))

    # Create z variables and constraints per flow
    z_vars: Dict[int, Dict[int, pulp.LpVariable]] = {}
    landing_reward_terms = []
    # Scoring constants matching postprocess
    SCORE_SCALE = 100.0
    LANDING_WEIGHT = 0.1
    total_volume_global = sum(f.volume for f in problem.flows) or 1.0

    for fid in cluster.flow_ids:
        y_list = flow_to_y.get(fid, [])
        if not y_list:
            continue
        kmax = len(y_list)
        z_dict: Dict[int, pulp.LpVariable] = {}
        for k in range(1, kmax + 1):
            z_dict[k] = pulp.LpVariable(f"z_{fid}_{k}", lowBound=0, upBound=1, cat="Binary")
        z_vars[fid] = z_dict
        # Exactly one k selected
        model += pulp.lpSum(z_dict[k] for k in range(1, kmax + 1)) == 1
        # Sum of y equals selected k
        sum_y = pulp.lpSum(y_vars[key] for key in y_list)
        model += pulp.lpSum(k * z_dict[k] for k in range(1, kmax + 1)) == sum_y
        # Landing reward contribution: weight by global flow weight
        flow_weight = (flow_lookup[fid].volume / total_volume_global) if total_volume_global > 0 else 0.0
        landing_reward = SCORE_SCALE * LANDING_WEIGHT * pulp.lpSum((1.0 / k) * z_dict[k] for k in range(1, kmax + 1))
        if flow_weight > 0:
            landing_reward_terms.append(flow_weight * landing_reward)

    # Approximate switch penalties across consecutive seconds (optional)
    # Build per-flow time index
    flow_times: Dict[int, List[int]] = {}
    for (fid, t), keys in flow_time.items():
        if fid not in flow_times:
            flow_times[fid] = []
        if t not in flow_times[fid]:
            flow_times[fid].append(t)
    for fid in flow_times:
        flow_times[fid].sort()
    # Map (fid, cell, t) -> has u var
    u_exists = set(u_vars.keys())
    switch_penalty_terms = []
    SWITCH_WEIGHT = 0.0  # disable switch penalty; rely on exact 1/k via z
    # For each consecutive pair of times, create s binary and link to differences where both sides exist
    for fid, times in flow_times.items():
        for i in range(1, len(times)):
            t0 = times[i - 1]
            t1 = times[i]
            s_var = pulp.LpVariable(f"switch_{fid}_{t1}", lowBound=0.0, upBound=1.0, cat="Binary")
            # For all cells appearing in both times, enforce s >= |u_t1 - u_t0|
            any_pair = False
            for cell in set(c for (_, c, tt) in u_exists if _ == fid and tt in (t0, t1)):
                k0 = (fid, cell, t0)
                k1 = (fid, cell, t1)
                if k0 in u_exists and k1 in u_exists:
                    any_pair = True
                    model += s_var >= u_vars[k1] - u_vars[k0]
                    model += s_var >= u_vars[k0] - u_vars[k1]
            if any_pair:
                switch_penalty_terms.append(SWITCH_WEIGHT * s_var)

    for (cell, time_slot), keys in cell_time.items():
        capacity = capacity_lookup(table, cell, time_slot)
        model += pulp.lpSum(x_vars[key] for key in keys) <= capacity

    objective_terms = []
    for key in x_keys:
        flow_id, cell, time_slot = key
        flow = flow_lookup[flow_id]
        reward = _x_reward(flow, cell, time_slot)
        objective_terms.append(reward * x_vars[key])

    # Weight per-flow objective terms to match global weighted scoring
    total_volume_global = sum(f.volume for f in problem.flows) or 1.0
    weighted_objective_terms = []
    for term, key in zip(objective_terms, x_keys):
        flow_id = key[0]
        flow_weight = (flow_lookup[flow_id].volume / total_volume_global)
        weighted_objective_terms.append(flow_weight * term)

    slack_penalties = [1e5 * drop_vars[fid] for fid in cluster.flow_ids]

    model += (
        pulp.lpSum(weighted_objective_terms)
        + pulp.lpSum(landing_reward_terms)
        - pulp.lpSum(slack_penalties)
        - pulp.lpSum(switch_penalty_terms)
    )

    # -----------------------------
    # Warm-start: seed variables
    # -----------------------------
    try:
        x_key_set = set(x_keys)
        y_key_set = set(y_keys)
        # Track totals per flow for slack and landings
        assigned_sum: Dict[int, float] = {fid: 0.0 for fid in cluster.flow_ids}
        y_activated: Dict[int, set] = {fid: set() for fid in cluster.flow_ids}
        # Helper to count landing episodes in warm-start
        def _episodes(segments: List[ScheduleSegment]) -> int:
            if not segments:
                return 0
            by_time: Dict[int, List[ScheduleSegment]] = {}
            for s in segments:
                by_time.setdefault(s.time, []).append(s)
            count = 0
            last = None
            for t in sorted(by_time.keys()):
                arr = by_time[t]
                rep = max(arr, key=lambda ss: ss.rate)
                cell = (rep.x, rep.y)
                if last is None or cell != last:
                    count += 1
                    last = cell
            return count
        for fid in cluster.flow_ids:
            for seg in warm_start.get(fid, []):
                key = (fid, (seg.x, seg.y), seg.time)
                if key in x_key_set:
                    # Seed x and activation u for used slots
                    x_vars[key].setInitialValue(seg.rate)
                    u_vars[key].setInitialValue(1)
                    assigned_sum[fid] += seg.rate
                    y_key = (fid, (seg.x, seg.y))
                    if y_key in y_key_set:
                        y_activated[fid].add(y_key)
        # Seed y variables and slack/drop/landing_extra
        for fid in cluster.flow_ids:
            # y activations
            for y_key in y_activated[fid]:
                y_vars[y_key].setInitialValue(1)
            # drop slack
            drop_guess = max(0.0, flow_lookup[fid].volume - assigned_sum[fid])
            drop_vars[fid].setInitialValue(drop_guess)
            # z warm-start for piecewise 1/k: select observed episodes if available
            warm_k = _episodes(warm_start.get(fid, []))
            if warm_k and fid in z_vars:
                for k, var in z_vars[fid].items():
                    var.setInitialValue(1 if k == warm_k else 0)
    except Exception:
        # Warm-start is best-effort; ignore if solver/variables don't accept it
        pass

    # Optionally fix flows that are already fully satisfied with a single landing episode in warm-start
    try:
        for fid in cluster.flow_ids:
            segments = warm_start.get(fid, [])
            if not segments:
                continue
            # Sum assigned and episode count
            total_rate = sum(s.rate for s in segments)
            episodes = 0
            last = None
            for t in sorted({s.time for s in segments}):
                batch = [s for s in segments if s.time == t]
                rep = max(batch, key=lambda ss: ss.rate)
                cell = (rep.x, rep.y)
                if last is None or cell != last:
                    episodes += 1
                    last = cell
            if episodes != 1:
                continue
            flow = flow_lookup[fid]
            if total_rate + 1e-6 < flow.volume:
                continue
            # Ensure all warm keys exist in this model (respect rolling window)
            warm_keys = {(fid, (s.x, s.y), s.time) for s in segments}
            x_key_set = set(x_keys)
            if not all(k in x_key_set for k in warm_keys):
                continue
            # Fix variables equal to warm assignment for this flow
            for key in x_keys:
                if key[0] != fid:
                    continue
                if key in warm_keys:
                    # Find matching segment rate
                    rx = next(s.rate for s in segments if (fid, (s.x, s.y), s.time) == key)
                    model += u_vars[key] == 1
                    model += x_vars[key] == rx
                else:
                    model += u_vars[key] == 0
                    model += x_vars[key] == 0
    except Exception:
        pass

    if hasattr(pulp, "HiGHS_CMD"):
        # Prefer HiGHS when available; allow more time and tighter gap
        try:
            solver = pulp.HiGHS_CMD(msg=False, timeLimit=10, options=["mip_rel_gap=0.0"])  # aim for exact
        except TypeError:
            # Fallback signature without options param
            solver = pulp.HiGHS_CMD(msg=False, timeLimit=10)
    else:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=10, gapRel=0.0)
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
    # Cluster-by-cluster with acceptance guard using local scorer
    ACCEPT_GUARD = True
    for cluster in clusters:
        # Candidate 1: MILP solution
        milp_solution = solve_cluster(cluster, problem, candidate_index, capacity_table, warm_start)
        cand1 = {fid: segs[:] for fid, segs in assignments.items()}
        for fid, segs in milp_solution.items():
            cand1.setdefault(fid, []).extend(segs)
        score1 = _compute_total_score(problem, cand1)

        # Candidate 2: warm-start fallback for this cluster
        fb_solution = _fallback(cluster, warm_start)
        cand2 = {fid: segs[:] for fid, segs in assignments.items()}
        for fid, segs in fb_solution.items():
            cand2.setdefault(fid, []).extend(segs)
        score2 = _compute_total_score(problem, cand2)

        # Accept the better one (or always MILP if guard disabled)
        chosen = cand1 if (not ACCEPT_GUARD or score1 >= score2) else cand2
        assignments = chosen

    # Post-process to reduce landing switches and improve delay
    assignments = _postprocess_consolidate_landings(problem, assignments, capacity_table)
    assignments = _postprocess_left_shift(problem, assignments, capacity_table)

    # Safety: enforce single landing cell per flow per time by keeping dominant segment
    assignments = _enforce_single_cell_per_time(assignments)

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
# Local scoring (matches problem_definition and src/postprocess)
# ---------------------------------------------------------------------------


def _compute_flow_scores(flow: Flow, segments: List[ScheduleSegment]) -> Tuple[float, float, float, float, float]:
    if not segments:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    by_time: Dict[int, List[ScheduleSegment]] = {}
    for s in segments:
        by_time.setdefault(s.time, []).append(s)

    total_sent = sum(s.rate for s in segments)
    throughput = min(1.0, total_sent / flow.volume) if flow.volume > 0 else 0.0

    T_MAX = 10.0
    delay = 0.0
    for t, arr in by_time.items():
        qsum = sum(s.rate for s in arr)
        delay += (T_MAX / (t + T_MAX)) * (qsum / flow.volume)

    alpha = 0.1
    distance = 0.0
    for arr in by_time.values():
        for s in arr:
            manhattan = abs(flow.access_x - s.x) + abs(flow.access_y - s.y)
            distance += (s.rate / flow.volume) * (2.0 ** (-alpha * manhattan))

    # landing episodes (increment when representative cell changes across time)
    k_seq = 0
    last_cell: Tuple[int, int] | None = None
    for t in sorted(by_time.keys()):
        arr = by_time[t]
        rep = max(arr, key=lambda ss: ss.rate)
        cell = (rep.x, rep.y)
        if last_cell is None or cell != last_cell:
            k_seq += 1
            last_cell = cell
    landing = 1.0 / max(1, k_seq)

    SCORE_SCALE = 100.0
    total = SCORE_SCALE * (0.4 * throughput + 0.2 * delay + 0.3 * distance + 0.1 * landing)
    return total, throughput, delay, distance, landing


def _compute_total_score(problem: Problem, assignments: Dict[int, List[ScheduleSegment]]) -> float:
    total_volume = sum(flow.volume for flow in problem.flows)
    if total_volume <= EPSILON:
        return 0.0
    total_score = 0.0
    for flow in problem.flows:
        segs = assignments.get(flow.flow_id, [])
        flow_score, *_ = _compute_flow_scores(flow, segs)
        weight = flow.volume / total_volume
        total_score += weight * flow_score
    return total_score


# ---------------------------------------------------------------------------
# Post-processing: consolidate landings to improve 1/k score
# ---------------------------------------------------------------------------


def _postprocess_consolidate_landings(
    problem: Problem,
    assignments: Dict[int, List[ScheduleSegment]],
    capacity_table: Dict[Tuple[int, int], List[float]],
) -> Dict[int, List[ScheduleSegment]]:
    # Build current usage per (cell, time)
    usage: Dict[Tuple[Tuple[int, int], int], float] = {}
    for segs in assignments.values():
        for seg in segs:
            key = ((seg.x, seg.y), seg.time)
            usage[key] = usage.get(key, 0.0) + seg.rate

    def capacity(cell: Tuple[int, int], t: int) -> float:
        return capacity_lookup(capacity_table, cell, t)

    new_assignments: Dict[int, List[ScheduleSegment]] = {}
    for flow in problem.flows:
        segs = [ScheduleSegment(s.time, s.x, s.y, s.rate) for s in assignments.get(flow.flow_id, [])]
        if not segs:
            new_assignments[flow.flow_id] = []
            continue
        # Determine dominant cell by total allocated rate
        totals: Dict[Tuple[int, int], float] = {}
        for s in segs:
            cell = (s.x, s.y)
            totals[cell] = totals.get(cell, 0.0) + s.rate
        anchor = max(totals.items(), key=lambda kv: kv[1])[0]

        # Index existing segments by time for merge
        by_time: Dict[int, List[ScheduleSegment]] = {}
        for s in segs:
            by_time.setdefault(s.time, []).append(s)

        times_sorted = sorted(by_time)
        ea_anchor = flow.earliest_arrival(anchor)
        for t in times_sorted:
            bucket = by_time[t]
            # If only one segment already and it's on anchor, nothing to do
            if len(bucket) == 1 and (bucket[0].x, bucket[0].y) == anchor:
                continue
            # Do not move into anchor before its earliest arrival
            if t < ea_anchor:
                continue
            # Find existing anchor segment if any
            anchor_idx = next((i for i, s in enumerate(bucket) if (s.x, s.y) == anchor), None)
            anchor_seg = bucket[anchor_idx] if anchor_idx is not None else None
            # Available free capacity on anchor at this time
            free = capacity(anchor, t) - usage.get((anchor, t), 0.0)
            if free <= 1e-9:
                continue
            # Move from non-anchor segments into anchor
            for i in range(len(bucket)):
                s = bucket[i]
                if (s.x, s.y) == anchor or s.rate <= 1e-9:
                    continue
                move = min(s.rate, free)
                if move <= 1e-9:
                    continue
                # Apply move
                s.rate -= move
                usage[((s.x, s.y), t)] = usage.get(((s.x, s.y), t), 0.0) - move
                free -= move
                if anchor_seg is not None:
                    usage[(anchor, t)] = usage.get((anchor, t), 0.0) + move
                    anchor_seg.rate += move
                if free <= 1e-9:
                    break
            # Cleanup empty segments
            new_bucket = [s for s in bucket if s.rate > 1e-9]
            # Enforce single segment per time by merging into anchor if any other remain and capacity allows
            if len(new_bucket) > 1 and anchor_seg is not None:
                others = [s for s in new_bucket if s is not anchor_seg]
                # Try to move the rest if any free left
                free = capacity(anchor, t) - usage.get((anchor, t), 0.0)
                for s in others:
                    if free <= 1e-9:
                        break
                    move = min(s.rate, free)
                    if move > 1e-9:
                        s.rate -= move
                        anchor_seg.rate += move
                        usage[((s.x, s.y), t)] = usage.get(((s.x, s.y), t), 0.0) - move
                        usage[(anchor, t)] = usage.get((anchor, t), 0.0) + move
                        free -= move
                reduced = [s for s in by_time[t] if s.rate > 1e-9]
                # If still more than one segment (capacity insufficient), revert to original bucket to keep validity
                if len(reduced) > 1:
                    by_time[t] = [s for s in bucket if s.rate > 1e-9]
                else:
                    by_time[t] = reduced
            else:
                by_time[t] = new_bucket

        # Rebuild segment list
        updated: List[ScheduleSegment] = []
        for t in by_time:
            updated.extend(by_time[t])
        updated.sort(key=lambda s: (s.time, s.x, s.y))
        new_assignments[flow.flow_id] = updated

    return new_assignments


def _postprocess_left_shift(
    problem: Problem,
    assignments: Dict[int, List[ScheduleSegment]],
    capacity_table: Dict[Tuple[int, int], List[float]],
) -> Dict[int, List[ScheduleSegment]]:
    usage: Dict[Tuple[Tuple[int, int], int], float] = {}
    for segs in assignments.values():
        for seg in segs:
            key = ((seg.x, seg.y), seg.time)
            usage[key] = usage.get(key, 0.0) + seg.rate

    def capacity(cell: Tuple[int, int], t: int) -> float:
        return capacity_lookup(capacity_table, cell, t)

    new_assignments: Dict[int, List[ScheduleSegment]] = {}
    for flow in problem.flows:
        segs = [ScheduleSegment(s.time, s.x, s.y, s.rate) for s in assignments.get(flow.flow_id, [])]
        if not segs:
            new_assignments[flow.flow_id] = []
            continue
        # Anchor cell: most allocated
        totals: Dict[Tuple[int, int], float] = {}
        for s in segs:
            cell = (s.x, s.y)
            totals[cell] = totals.get(cell, 0.0) + s.rate
        anchor = max(totals.items(), key=lambda kv: kv[1])[0]
        ea = flow.earliest_arrival(anchor)

        # Index by time on anchor and non-anchor
        by_time: Dict[int, List[ScheduleSegment]] = {}
        for s in segs:
            by_time.setdefault(s.time, []).append(s)

        times_sorted = sorted(by_time)
        for t in times_sorted:
            bucket = by_time[t]
            # Work only with anchor segments for shifting earlier
            for s in list(bucket):
                if (s.x, s.y) != anchor or s.rate <= 1e-9:
                    continue
                if t <= ea:
                    continue
                # Try to move left into earliest available slots
                remain = s.rate
                moved_total = 0.0
                tau = ea
                while remain > 1e-9 and tau < t:
                    free = capacity(anchor, tau) - usage.get((anchor, tau), 0.0)
                    if free > 1e-9:
                        mv = min(remain, free)
                        remain -= mv
                        moved_total += mv
                        usage[(anchor, tau)] = usage.get((anchor, tau), 0.0) + mv
                        # Insert/merge segment at tau
                        existing_list = by_time.setdefault(tau, [])
                        existing_anchor = next((seg for seg in existing_list if (seg.x, seg.y) == anchor), None)
                        if existing_anchor is None:
                            existing_list.append(ScheduleSegment(time=tau, x=anchor[0], y=anchor[1], rate=mv))
                        else:
                            existing_anchor.rate += mv
                    tau += 1
                if moved_total > 1e-9:
                    s.rate -= moved_total
                    usage[(anchor, t)] = usage.get((anchor, t), 0.0) - moved_total
        # Cleanup empty segments and sort
        updated: List[ScheduleSegment] = []
        for t, arr in by_time.items():
            for seg in arr:
                if seg.rate > 1e-9:
                    updated.append(seg)
        updated.sort(key=lambda s: (s.time, s.x, s.y))
        new_assignments[flow.flow_id] = updated

    return new_assignments


def _enforce_single_cell_per_time(assignments: Dict[int, List[ScheduleSegment]]) -> Dict[int, List[ScheduleSegment]]:
    fixed: Dict[int, List[ScheduleSegment]] = {}
    for fid, segs in assignments.items():
        by_time: Dict[int, List[ScheduleSegment]] = {}
        for s in segs:
            by_time.setdefault(s.time, []).append(s)
        new_list: List[ScheduleSegment] = []
        for t, arr in by_time.items():
            if len(arr) == 1:
                new_list.append(arr[0])
                continue
            # Keep the segment with highest rate; drop others if on different cells
            arr.sort(key=lambda s: s.rate, reverse=True)
            top = arr[0]
            # Merge any other segments on the same cell into top
            for other in arr[1:]:
                if (other.x, other.y) == (top.x, top.y):
                    top.rate += other.rate
            new_list.append(ScheduleSegment(time=t, x=top.x, y=top.y, rate=top.rate))
        new_list.sort(key=lambda s: (s.time, s.x, s.y))
        fixed[fid] = new_list
    return fixed


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
