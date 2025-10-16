from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .bandwidth import capacities_for_t
from .model import Coord, Flow, Problem, manhattan


EPS = 1e-9


@dataclass(order=True)
class _QueueEntry:
    priority: float
    time: int
    fid: int
    coord_hint: Coord


class TemporalGraphScheduler:
    """Receding-horizon scheduler inspired by the NTN deterministic routing strategy.

    For each time step, it builds a small time-expanded graph (window of size `horizon`)
    with the current residual capacities, scores candidate (time, landing) pairs,
    and greedily allocates using a global priority queue. Only the decisions for the
    current time are committed, while future slots are re-planned at the next step.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        top_k: int = 12,
        horizon: int = 12,
        stickiness_bonus: float = 0.08,
        reuse_bonus: float = 0.04,
        new_landing_penalty: float = 0.06,
        tmax_delay: int = 10,
        allow_full_scan: bool = True,
    ):
        self.alpha = alpha
        self.top_k = top_k
        self.horizon = horizon
        self.stickiness_bonus = stickiness_bonus
        self.reuse_bonus = reuse_bonus
        self.new_landing_penalty = new_landing_penalty
        self.tmax_delay = tmax_delay
        self.allow_full_scan = allow_full_scan
        self._candidates: Dict[int, List[Tuple[Coord, float]]] = {}

    def _precompute_candidates(self, problem: Problem) -> None:
        if self._candidates:
            return
        for flow in problem.flows:
            acc = flow.access_coord
            cands: List[Tuple[Coord, float]] = []
            for x in range(flow.m1, flow.m2 + 1):
                for y in range(flow.n1, flow.n2 + 1):
                    uav = problem.uavs[(x, y)]
                    d = manhattan(acc, (x, y))
                    score = (0.7 * uav.B) * (2.0 ** (-self.alpha * d))
                    cands.append(((x, y), score))
            cands.sort(key=lambda it: it[1], reverse=True)
            self._candidates[flow.fid] = cands[: self.top_k]

    def _base_value(self, flow: Flow, coord: Coord, time: int) -> float:
        delay_weight = self.tmax_delay / (time + self.tmax_delay)
        d = manhattan(flow.access_coord, coord)
        dist_factor = 2.0 ** (-self.alpha * d)
        value = 0.4 + 0.2 * delay_weight + 0.3 * dist_factor
        if flow.last_landing is not None and coord == flow.last_landing:
            value += self.stickiness_bonus
        if coord in flow.landings_used:
            value += self.reuse_bonus
        else:
            penalty = self.new_landing_penalty / (len(flow.landings_used) + 1)
            value -= penalty
        return value

    def _best_candidate(
        self,
        flow: Flow,
        time: int,
        caps: Dict[int, Dict[Coord, float]],
    ) -> Tuple[Optional[Coord], float]:
        best_coord: Optional[Coord] = None
        best_value = float("-inf")

        for coord, _score in self._candidates.get(flow.fid, []):
            cap = caps[time].get(coord, 0.0)
            if cap <= EPS:
                continue
            value = self._base_value(flow, coord, time)
            if value > best_value:
                best_value = value
                best_coord = coord

        if best_coord is None and self.allow_full_scan:
            for x in range(flow.m1, flow.m2 + 1):
                for y in range(flow.n1, flow.n2 + 1):
                    coord = (x, y)
                    cap = caps[time].get(coord, 0.0)
                    if cap <= EPS:
                        continue
                    value = self._base_value(flow, coord, time)
                    if value > best_value:
                        best_value = value
                        best_coord = coord

        if best_coord is None:
            return None, 0.0
        return best_coord, best_value

    def schedule(self, problem: Problem) -> Dict[int, List[Tuple[int, int, int, float]]]:
        self._precompute_candidates(problem)

        schedules: Dict[int, List[Tuple[int, int, int, float]]] = {f.fid: [] for f in problem.flows}
        flow_state: Dict[int, Flow] = {f.fid: f for f in problem.flows}
        caps_by_t: Dict[int, Dict[Coord, float]] = {
            t: capacities_for_t(problem.uavs, t) for t in range(problem.T)
        }

        for current_t in range(problem.T):
            window_end = min(problem.T, current_t + self.horizon)
            # Collect flows that could use capacity within the window
            window_flows = [
                f for f in flow_state.values()
                if f.remaining > EPS and f.t_start < window_end
            ]
            if not window_flows:
                continue

            # Local copies for planning
            local_caps: Dict[int, Dict[Coord, float]] = {
                t: caps_by_t[t].copy() for t in range(current_t, window_end)
            }
            remaining_tmp: Dict[int, float] = {f.fid: f.remaining for f in window_flows}
            assignments: Dict[Tuple[int, int], Tuple[Coord, float]] = {}

            heap: List[_QueueEntry] = []
            for flow in window_flows:
                start_time = max(current_t, flow.t_start)
                for t in range(start_time, window_end):
                    coord, value = self._best_candidate(flow, t, local_caps)
                    if coord is None or value <= EPS:
                        continue
                    heapq.heappush(
                        heap,
                        _QueueEntry(priority=-value, time=t, fid=flow.fid, coord_hint=coord),
                    )

            while heap:
                entry = heapq.heappop(heap)
                flow = flow_state[entry.fid]
                if remaining_tmp[flow.fid] <= EPS:
                    continue
                if entry.time < flow.t_start:
                    continue
                if entry.time < current_t:
                    continue
                if (flow.fid, entry.time) in assignments:
                    continue

                coord, value = self._best_candidate(flow, entry.time, local_caps)
                if coord is None or value <= EPS:
                    continue
                expected_priority = -entry.priority
                if coord != entry.coord_hint or abs(value - expected_priority) > 1e-6:
                    heapq.heappush(
                        heap,
                        _QueueEntry(priority=-value, time=entry.time, fid=flow.fid, coord_hint=coord),
                    )
                    continue

                avail = local_caps[entry.time].get(coord, 0.0)
                if avail <= EPS:
                    continue

                send = min(remaining_tmp[flow.fid], avail)
                if send <= EPS:
                    continue

                assignments[(flow.fid, entry.time)] = (coord, send)
                remaining_tmp[flow.fid] -= send
                local_caps[entry.time][coord] = max(0.0, avail - send)

            # Commit decisions for the current time only
            for (fid, time), (coord, amount) in assignments.items():
                if time != current_t or amount <= EPS:
                    continue
                flow = flow_state[fid]
                schedules[fid].append((time, coord[0], coord[1], amount))
                flow.remaining = max(0.0, flow.remaining - amount)
                # Update landing usage metadata
                if coord not in flow.landings_used:
                    flow.landings_used.add(coord)
                flow.last_landing = coord
                flow.unique_landings = len(flow.landings_used)
                # Reduce global capacity for this slot
                prev_cap = caps_by_t[time].get(coord, 0.0)
                caps_by_t[time][coord] = max(0.0, prev_cap - amount)

        for fid in schedules:
            schedules[fid].sort()
        return schedules
