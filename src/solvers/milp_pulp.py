from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import pulp

from ..bandwidth import capacities_for_t
from ..model import Problem, Flow, Coord, manhattan


def _distance_factor(alpha: float, d: int) -> float:
    return 2.0 ** (-alpha * d)


def _precompute_candidates(problem: Problem, alpha: float, top_k: int) -> Dict[int, List[Tuple[Coord, float]]]:
    candidates: Dict[int, List[Tuple[Coord, float]]] = {}
    for f in problem.flows:
        acc = (f.ax, f.ay)
        cands: List[Tuple[Coord, float]] = []
        for x in range(f.m1, f.m2 + 1):
            for y in range(f.n1, f.n2 + 1):
                u = problem.uavs[(x, y)]
                d = manhattan(acc, (x, y))
                score = (0.7 * u.B) * _distance_factor(alpha, d)
                cands.append(((x, y), score))
        cands.sort(key=lambda it: it[1], reverse=True)
        candidates[f.fid] = cands[: top_k]
    return candidates


def solve_milp(
    problem: Problem,
    *,
    alpha: float = 0.1,
    top_k: int = 30,
    tmax_delay: int = 10,
    time_window: Optional[int] = None,
    solver: Optional[pulp.LpSolver] = None,
) -> Dict[int, List[Tuple[int, int, int, float]]]:
    """MILP implementation using PuLP (CBC by default if available).

    Returns schedules as fid -> list of (t, x, y, z) with z > 0.
    """

    # Precompute caps per time and candidate sets
    caps_by_t = {t: capacities_for_t(problem.uavs, t) for t in range(problem.T)}
    candidates = _precompute_candidates(problem, alpha, top_k)

    total_Q = sum(f.total for f in problem.flows) or 1.0

    # Build model
    m = pulp.LpProblem("UAV_U2GL_Scheduling", pulp.LpMaximize)

    # Variables
    z_vars: Dict[Tuple[int, int, Tuple[int, int]], pulp.LpVariable] = {}
    y_vars: Dict[Tuple[int, int, Tuple[int, int]], pulp.LpVariable] = {}
    u_vars: Dict[Tuple[int, Tuple[int, int]], pulp.LpVariable] = {}
    s_vars: Dict[int, pulp.LpVariable] = {}
    v_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}

    # Decision variables domains
    for f in problem.flows:
        s_vars[f.fid] = pulp.LpVariable(f"s_{f.fid}", lowBound=0.0, upBound=1.0)
        # Landing used binaries
        for (coord, _) in candidates[f.fid]:
            u_vars[(f.fid, coord)] = pulp.LpVariable(f"u_{f.fid}_{coord[0]}_{coord[1]}", cat="Binary")
        # v_{f,k} for k=1..Kmax where Kmax is number of candidates
        Kmax = max(1, len(candidates[f.fid]))
        for k in range(1, Kmax + 1):
            v_vars[(f.fid, k)] = pulp.LpVariable(f"v_{f.fid}_{k}", cat="Binary")

        # z and y for feasible (t >= t_start) and candidate coords with non-zero capacity possibly
        last_t = problem.T - 1 if time_window is None else min(problem.T - 1, f.t_start + time_window)
        for t in range(f.t_start, last_t + 1):
            for (coord, _) in candidates[f.fid]:
                # If capacity at (t, coord) is always zero, skip
                if caps_by_t[t].get(coord, 0.0) <= 0.0:
                    continue
                z_vars[(f.fid, t, coord)] = pulp.LpVariable(f"z_{f.fid}_{t}_{coord[0]}_{coord[1]}", lowBound=0.0)
                y_vars[(f.fid, t, coord)] = pulp.LpVariable(f"y_{f.fid}_{t}_{coord[0]}_{coord[1]}", cat="Binary")

    # Constraints
    # 1) Capacity per (t, coord): sum_f z_{f,t,coord} <= cap_{coord,t}
    for t in range(problem.T):
        caps = caps_by_t[t]
        # consider only coords that appear in z_vars for this t
        coords_at_t = set(coord for (fid, tt, coord) in z_vars.keys() if tt == t)
        for coord in coords_at_t:
            m += (
                pulp.lpSum(z_vars[(fid, t, coord)] for (fid, tt, cc) in z_vars.keys() if tt == t and cc == coord)
                <= caps.get(coord, 0.0)
            ), f"cap_t{t}_x{coord[0]}_y{coord[1]}"

    for f in problem.flows:
        # 2) Flow volume: sum_t,j z_{f,t,j} <= Q_f
        m += (
            pulp.lpSum(z for (fid, t, coord), z in z_vars.items() if fid == f.fid) <= f.total
        ), f"flow_budget_f{f.fid}"

        # 3) Link s_f <= (sum z) / Q_f and s_f <= 1 handled by bounds
        m += (
            s_vars[f.fid] * f.total <= pulp.lpSum(z for (fid, t, coord), z in z_vars.items() if fid == f.fid)
        ), f"s_link_f{f.fid}"

        # 4) One landing per time: sum_j y_{f,t,j} <= 1 for each feasible t
        feasible_ts = sorted({t for (fid, t, coord) in z_vars.keys() if fid == f.fid})
        for t in feasible_ts:
            m += (
                pulp.lpSum(y_vars[(fid, t, coord)] for (fid, tt, coord) in y_vars.keys() if fid == f.fid and tt == t) <= 1
            ), f"one_landing_f{f.fid}_t{t}"

        # 5) Link z to y: z_{f,t,j} <= Q_f * y_{f,t,j}
        for (fid, t, coord), z in list(z_vars.items()):
            if fid != f.fid:
                continue
            m += (z <= f.total * y_vars[(fid, t, coord)]), f"link_zy_f{fid}_t{t}_{coord}"

        # 6) Landing usage link: z_{f,t,j} <= Q_f * u_{f,j}
        for (fid, t, coord), z in list(z_vars.items()):
            if fid != f.fid:
                continue
            m += (z <= f.total * u_vars[(f.fid, coord)]), f"link_zu_f{fid}_t{t}_{coord}"

        # 7) Landing count linearization: sum_j u_{f,j} = sum_k k * v_{f,k}; sum_k v_{f,k} = 1
        Kmax = max(1, len(candidates[f.fid]))
        m += (
            pulp.lpSum(u_vars[(f.fid, coord)] for (coord, _) in candidates[f.fid])
            == pulp.lpSum(k * v_vars[(f.fid, k)] for k in range(1, Kmax + 1))
        ), f"count_link_f{f.fid}"
        m += (
            pulp.lpSum(v_vars[(f.fid, k)] for k in range(1, Kmax + 1)) == 1
        ), f"count_onehot_f{f.fid}"

    # Objective
    # For each flow, weight by Q_f / total_Q
    obj_terms: List = []
    for f in problem.flows:
        w_flow = f.total / total_Q
        # Traffic completeness: 0.4 * s_f
        obj_terms.append(0.4 * w_flow * s_vars[f.fid])
        # Delay: 0.2 * sum_{t,j} w_delay(t) * z / Q_f
        for (fid, t, coord), z in z_vars.items():
            if fid != f.fid:
                continue
            w_delay = tmax_delay / (t + tmax_delay)
            obj_terms.append(0.2 * w_flow * (w_delay / f.total) * z)
        # Distance: 0.3 * sum_{t,j} w_dist * z / Q_f
        acc = (f.ax, f.ay)
        for (fid, t, coord), z in z_vars.items():
            if fid != f.fid:
                continue
            d = manhattan(acc, coord)
            w_dist = _distance_factor(alpha, d)
            obj_terms.append(0.3 * w_flow * (w_dist / f.total) * z)
        # Landing 1/k: 0.1 * sum_k (1/k) * v_{f,k}
        Kmax = max(1, len(candidates[f.fid]))
        for k in range(1, Kmax + 1):
            obj_terms.append(0.1 * w_flow * (1.0 / k) * v_vars[(f.fid, k)])

    m += pulp.lpSum(obj_terms) * 100.0

    # Solve
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False)
    m.solve(solver)

    # Extract solution
    schedules: Dict[int, List[Tuple[int, int, int, float]] ] = {f.fid: [] for f in problem.flows}
    for (fid, t, coord), z in z_vars.items():
        val = z.value() or 0.0
        if val > 1e-9:
            schedules[fid].append((t, coord[0], coord[1], float(val)))
    # Sort by time within each flow
    for fid in schedules:
        schedules[fid].sort()
    return schedules

