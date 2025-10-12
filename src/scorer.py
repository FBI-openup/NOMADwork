from __future__ import annotations

import sys
from typing import Dict, List, Tuple

from .io import parse_stream, parse_file
from .model import Problem, manhattan
from .bandwidth import capacities_for_t


def load_schedule(path: str) -> Dict[int, List[Tuple[int, int, int, float]]]:
    schedules: Dict[int, List[Tuple[int, int, int, float]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i = 0
    while i < len(lines):
        header = lines[i].split()
        fid = int(header[0])
        p = int(header[1])
        i += 1
        entries: List[Tuple[int, int, int, float]] = []
        for _ in range(p):
            t, x, y, z = lines[i].split()
            entries.append((int(t), int(x), int(y), float(z)))
            i += 1
        schedules[fid] = entries
    return schedules


def score(problem: Problem, schedules: Dict[int, List[Tuple[int, int, int, float]]]) -> float:
    # Validate capacity feasibility and compute per-flow scores
    total_weight = sum(f.total for f in problem.flows)
    if total_weight <= 0:
        return 0.0

    # Build per-time UAV usage from schedules to check capacity
    # usage[(t, (x,y))] = sum z
    usage: Dict[Tuple[int, Tuple[int, int]], float] = {}
    for fid, entries in schedules.items():
        for t, x, y, z in entries:
            usage[(t, (x, y))] = usage.get((t, (x, y)), 0.0) + z

    # Capacity violations count (not fatal, just noted during scoring)
    violations = 0
    for t in range(problem.T):
        caps = capacities_for_t(problem.uavs, t)
        for coord, cap in caps.items():
            used = usage.get((t, coord), 0.0)
            if used - cap > 1e-6:
                violations += 1

    # Per-flow score
    alpha = 0.1
    T_max = 10
    total_score = 0.0
    for f in problem.flows:
        entries = sorted(schedules.get(f.fid, []))
        if not entries:
            continue
        # Aggregate by time to compute q_i at each t
        q_by_t: Dict[int, List[Tuple[Tuple[int, int], float]]] = {}
        for t, x, y, z in entries:
            q_by_t.setdefault(t, []).append(((x, y), z))

        q_total_sent = sum(z for arr in q_by_t.values() for (_, z) in arr)

        # 1. Traffic completeness
        traffic_score = min(1.0, q_total_sent / f.total) if f.total > 0 else 0.0

        # 2. Delay score
        delay_score = 0.0
        for t, arr in q_by_t.items():
            # Each second’s contribution based on earliest t_i seen at that second
            q_sum = sum(z for (_, z) in arr)
            delay_score += (T_max / (t + T_max)) * (q_sum / f.total)

        # 3. Distance score
        distance_score = 0.0
        for t, arr in q_by_t.items():
            for (x, y), z in arr:
                d = manhattan((f.ax, f.ay), (x, y))
                distance_score += (z / f.total) * (2 ** (-alpha * d))

        # 4. Landing UAV point score
        unique_landings = []
        last = None
        for t in sorted(q_by_t.keys()):
            # Decide the representative landing at t as the (x,y) with max z
            arr = sorted(q_by_t[t], key=lambda it: it[1], reverse=True)
            rep = arr[0][0]
            if last != rep:
                unique_landings.append(rep)
                last = rep
        k = max(1, len(unique_landings))
        landing_point_score = 1.0 / k

        flow_score = 100.0 * (
            0.4 * traffic_score + 0.2 * delay_score + 0.3 * distance_score + 0.1 * landing_point_score
        )
        total_score += (f.total / total_weight) * flow_score

    return total_score


def main() -> None:
    # Usage modes:
    # 1) stdin problem + schedule path: python -m src.scorer <schedule>
    # 2) problem path + schedule path:  python -m src.scorer <problem> <schedule>
    if len(sys.argv) == 2:
        problem = parse_stream(sys.stdin)
        schedule_path = sys.argv[1]
    elif len(sys.argv) == 3:
        problem = parse_file(sys.argv[1])
        schedule_path = sys.argv[2]
    else:
        print("Usage: python -m src.scorer <schedule>  OR  python -m src.scorer <problem> <schedule>", file=sys.stderr)
        sys.exit(2)
    schedules = load_schedule(schedule_path)
    s = score(problem, schedules)
    print(f"Total Score: {s:.3f}")


if __name__ == "__main__":
    main()
