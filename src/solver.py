"""Main solver orchestration and MILP builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import pulp
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without PuLP
    pulp = None  # type: ignore[assignment]

from .capacity import build_capacity_table, capacity_lookup
from .candidates import CandidateConfig, CandidateIndex, FlowCandidate, build_candidate_index
from .clustering import Cluster, cluster_flows
from .heuristics import WarmStart, build_warm_start
from .models import Flow, Problem, ScheduleSegment, ScoreSummary
from .postprocess import compute_score, merge_assignments, normalize_assignments

SlotKey = Tuple[int, Tuple[int, int], int]


@dataclass(slots=True)
class SolverConfig:
    total_time_limit: float = 5.0
    cluster_time_floor: float = 0.5
    slack_penalty: float = 1e5
    landing_penalty_scale: float = 1.0
    mip_gap: float = 0.01


@dataclass(slots=True)
class SolverResult:
    assignments: Dict[int, List[ScheduleSegment]]
    score: ScoreSummary
    clusters: List[Cluster]


def _fallback_from_warm_start(cluster: Cluster, warm_start: WarmStart) -> Dict[int, List[ScheduleSegment]]:
    assignments: Dict[int, List[ScheduleSegment]] = {}
    for flow_id in cluster.flow_ids:
        segments = [
            ScheduleSegment(time=segment.time, x=segment.x, y=segment.y, rate=segment.rate)
            for segment in warm_start.assignments.get(flow_id, [])
        ]
        assignments[flow_id] = segments
    return assignments


def solve_problem(
    problem: Problem,
    solver_config: SolverConfig | None = None,
    candidate_config: CandidateConfig | None = None,
) -> SolverResult:
    solver_config = solver_config or SolverConfig()
    candidate_config = candidate_config or CandidateConfig()

    capacity_table = build_capacity_table(problem)
    candidate_index = build_candidate_index(problem, capacity_table, candidate_config)
    clusters = cluster_flows(candidate_index)
    warm_start = build_warm_start(problem, candidate_index, capacity_table)

    assignments: Dict[int, List[ScheduleSegment]] = {}

    total_flows = len(problem.flows)
    remaining_time_budget = solver_config.total_time_limit

    for cluster in clusters:
        share = len(cluster.flow_ids) / total_flows if total_flows else 0.0
        requested_time = max(
            solver_config.cluster_time_floor,
            share * solver_config.total_time_limit,
        )
        if remaining_time_budget is not None:
            requested_time = min(requested_time, remaining_time_budget)
            remaining_time_budget = max(0.0, remaining_time_budget - requested_time)

        cluster_solution = _solve_cluster(
            cluster=cluster,
            problem=problem,
            candidate_index=candidate_index,
            capacity_table=capacity_table,
            solver_config=solver_config,
            warm_start=warm_start,
            time_limit=requested_time,
        )
        merge_assignments(assignments, cluster_solution)

    normalize_assignments(assignments)
    score = compute_score(problem, assignments)
    return SolverResult(assignments=assignments, score=score, clusters=clusters)


def _build_indices(cluster: Cluster, index: CandidateIndex):
    x_keys: List[SlotKey] = []
    flow_time: Dict[Tuple[int, int], List[SlotKey]] = {}
    cell_time: Dict[Tuple[Tuple[int, int], int], List[SlotKey]] = {}
    flow_cell: Dict[Tuple[int, Tuple[int, int]], List[SlotKey]] = {}
    y_keys: List[Tuple[int, Tuple[int, int]]] = []
    y_key_seen: set[Tuple[int, Tuple[int, int]]] = set()

    for flow_id in cluster.flow_ids:
        candidate = index.flows.get(flow_id)
        if candidate is None:
            continue
        for cell, slots in candidate.time_slots.items():
            for time_slot in slots:
                key = (flow_id, cell, time_slot)
                x_keys.append(key)
                flow_time.setdefault((flow_id, time_slot), []).append(key)
                cell_time.setdefault((cell, time_slot), []).append(key)
                flow_cell.setdefault((flow_id, cell), []).append(key)
                if (flow_id, cell) not in y_key_seen:
                    y_key_seen.add((flow_id, cell))
                    y_keys.append((flow_id, cell))
    return x_keys, flow_time, cell_time, flow_cell, y_keys


def _x_reward(flow: Flow, cell: Tuple[int, int], time_slot: int) -> float:
    throughput_component = 100.0 * 0.4 / max(flow.volume, 1e-6)
    delay_component = 100.0 * 0.2 * (10.0 / (time_slot + 10.0)) / max(flow.volume, 1e-6)
    distance = flow.travel_time(cell)
    distance_component = 100.0 * 0.3 * (2.0 ** (-0.1 * distance)) / max(flow.volume, 1e-6)
    return throughput_component + delay_component + distance_component


def _landing_penalty(flow_candidate: FlowCandidate, landing_penalty_scale: float) -> float:
    base = 100.0 * 0.1 / max(1, len(flow_candidate.time_slots))
    return landing_penalty_scale * base


def _solve_cluster(
    cluster: Cluster,
    problem: Problem,
    candidate_index: CandidateIndex,
    capacity_table,
    solver_config: SolverConfig,
    warm_start: WarmStart,
    time_limit: float,
) -> Dict[int, List[ScheduleSegment]]:
    if pulp is None:
        return _fallback_from_warm_start(cluster, warm_start)

    x_keys, flow_time, cell_time, flow_cell, y_keys = _build_indices(cluster, candidate_index)
    flow_lookup = {flow.flow_id: flow for flow in problem.flows if flow.flow_id in cluster.flow_ids}

    model = pulp.LpProblem("cluster", pulp.LpMaximize)

    x_vars = pulp.LpVariable.dicts("x", x_keys, lowBound=0.0)
    u_vars = pulp.LpVariable.dicts("u", x_keys, lowBound=0.0, upBound=1.0, cat="Binary")
    y_vars = pulp.LpVariable.dicts("y", y_keys, lowBound=0.0, upBound=1.0, cat="Binary")
    drop_vars = {
        flow_id: pulp.LpVariable(f"drop_{flow_id}", lowBound=0.0)
        for flow_id in cluster.flow_ids
    }

    # Capacity activation: x <= capacity * u
    for key in x_keys:
        flow_id, cell, time_slot = key
        capacity = capacity_lookup(capacity_table, cell, time_slot)
        model += x_vars[key] <= capacity * u_vars[key]

    # One landing cell per flow per time slot.
    for (flow_id, time_slot), keys in flow_time.items():
        model += pulp.lpSum(u_vars[key] for key in keys) <= 1

    # Flow completion with slack.
    for flow_id in cluster.flow_ids:
        x_terms = [
            x_vars[key]
            for key in x_keys
            if key[0] == flow_id
        ]
        model += pulp.lpSum(x_terms) + drop_vars[flow_id] == flow_lookup[flow_id].volume

    # Link y with u (landing activation).
    for (flow_id, cell), keys in flow_cell.items():
        if not keys:
            continue
        model += y_vars[(flow_id, cell)] <= pulp.lpSum(u_vars[key] for key in keys)
        for key in keys:
            model += u_vars[key] <= y_vars[(flow_id, cell)]

    # Per-cell capacity.
    for (cell, time_slot), keys in cell_time.items():
        capacity = capacity_lookup(capacity_table, cell, time_slot)
        model += pulp.lpSum(x_vars[key] for key in keys) <= capacity

    # Objective components.
    objective_terms = []
    for key in x_keys:
        flow_id, cell, time_slot = key
        flow = flow_lookup[flow_id]
        reward = _x_reward(flow, cell, time_slot)
        objective_terms.append(reward * x_vars[key])

    landing_penalty_terms = []
    for flow_id, cell in y_keys:
        candidate = candidate_index.flows[flow_id]
        penalty = _landing_penalty(candidate, solver_config.landing_penalty_scale)
        landing_penalty_terms.append(penalty * y_vars[(flow_id, cell)])

    slack_penalty_terms = [solver_config.slack_penalty * drop_vars[flow_id] for flow_id in cluster.flow_ids]

    model += pulp.lpSum(objective_terms) - pulp.lpSum(landing_penalty_terms) - pulp.lpSum(slack_penalty_terms)

    solver = pulp.PULP_CBC_CMD(
        msg=False,
        timeLimit=max(time_limit, solver_config.cluster_time_floor),
        gapRel=solver_config.mip_gap if solver_config.mip_gap > 0 else None,
    )
    model.solve(solver)

    status = pulp.LpStatus[model.status]
    assignments: Dict[int, List[ScheduleSegment]] = {flow_id: [] for flow_id in cluster.flow_ids}

    if status not in {"Optimal", "Feasible"}:
        # Fallback to warm-start assignments for these flows.
        for flow_id in cluster.flow_ids:
            warm_segments = warm_start.assignments.get(flow_id, [])
            assignments[flow_id] = [ScheduleSegment(seg.time, seg.x, seg.y, seg.rate) for seg in warm_segments]
        return assignments

    for key in x_keys:
        value = x_vars[key].value() or 0.0
        if value <= 1e-6:
            continue
        flow_id, cell, time_slot = key
        assignments.setdefault(flow_id, []).append(
            ScheduleSegment(time=time_slot, x=cell[0], y=cell[1], rate=value)
        )

    return assignments
