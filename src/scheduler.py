from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from .bandwidth import capacities_for_t
from .model import Flow, Problem, Coord, manhattan


class GreedyScheduler:
    def __init__(self, alpha: float = 0.1, top_k: int = 30, stickiness_bonus: float = 0.05, tmax_delay: int = 10):
        self.alpha = alpha
        self.top_k = top_k
        self.stickiness_bonus = stickiness_bonus
        self.tmax_delay = tmax_delay
        # Precomputed: for each flow id -> list of candidate coords sorted by static attractiveness
        self._candidates: Dict[int, List[Tuple[Coord, float]]] = {}

    def _distance_factor(self, d: int) -> float:
        # 2^{-alpha * d}
        return 2.0 ** (-self.alpha * d)

    def _delay_weight(self, t_since_start: int) -> float:
        # T_max / (t + T_max)
        return self.tmax_delay / (t_since_start + self.tmax_delay)

    def _precompute_candidates(self, problem: Problem) -> None:
        if self._candidates:
            return
        for f in problem.flows:
            acc = f.access_coord
            cands: List[Tuple[Coord, float]] = []
            for x in range(f.m1, f.m2 + 1):
                for y in range(f.n1, f.n2 + 1):
                    u = problem.uavs[(x, y)]
                    d = manhattan(acc, (x, y))
                    # Static attractiveness ~ avg capacity (0.7*B) times distance factor
                    score = (0.7 * u.B) * self._distance_factor(d)
                    cands.append(((x, y), score))
            # Sort descending by score and truncate
            cands.sort(key=lambda it: it[1], reverse=True)
            self._candidates[f.fid] = cands[: self.top_k]

    def schedule(self, problem: Problem) -> Dict[int, List[Tuple[int, int, int, float]]]:
        self._precompute_candidates(problem)
        schedules: Dict[int, List[Tuple[int, int, int, float]]] = {f.fid: [] for f in problem.flows}

        # Work on copies of flow state
        flow_state: Dict[int, Flow] = {f.fid: f for f in problem.flows}

        for t in range(problem.T):
            caps = capacities_for_t(problem.uavs, t)

            # Active flows that have started and have remaining > 0
            active: List[Flow] = [f for f in flow_state.values() if f.t_start <= t and f.remaining > 0.0]
            if not active:
                continue

            # Sort flows by a priority that favors earlier start, larger remaining, and higher delay weight
            def priority_key(f: Flow):
                t_since = max(0, t - f.t_start)
                delay_w = self._delay_weight(t_since)
                return (
                    -delay_w,  # higher first (so negative for ascending sort)
                    -(f.remaining / f.total),  # higher fraction remaining first
                    f.t_start,  # earlier starts first
                )

            active.sort(key=priority_key)

            for f in active:
                if f.remaining <= 0:
                    continue

                # Build dynamic candidate list: start with top-K precomputed, optionally extend with last landing
                cands = self._candidates.get(f.fid, [])
                # Include last landing if within range and not already present
                dyn_cands: List[Coord] = [c for (c, _) in cands]
                if f.last_landing is not None:
                    ll = f.last_landing
                    if f.landing_in_range(ll[0], ll[1]) and ll not in dyn_cands:
                        dyn_cands = [ll] + dyn_cands

                # Choose best candidate with capacity > 0
                best_coord: Optional[Coord] = None
                best_util = -1.0
                acc = f.access_coord
                for coord in dyn_cands:
                    cap = caps.get(coord, 0.0)
                    if cap <= 0.0:
                        continue
                    d = manhattan(acc, coord)
                    util = cap * self._distance_factor(d)
                    if f.last_landing is not None and coord == f.last_landing:
                        util += self.stickiness_bonus * cap
                    if util > best_util:
                        best_util = util
                        best_coord = coord

                if best_coord is None:
                    # No capacity available among candidates; try to opportunistically search full range (fallback)
                    for x in range(f.m1, f.m2 + 1):
                        for y in range(f.n1, f.n2 + 1):
                            cap = caps.get((x, y), 0.0)
                            if cap <= 0.0:
                                continue
                            d = manhattan(acc, (x, y))
                            util = cap * self._distance_factor(d)
                            if util > best_util:
                                best_util = util
                                best_coord = (x, y)

                if best_coord is None:
                    # Nothing to send this second for this flow
                    continue

                # Allocate up to remaining and available capacity
                avail = caps[best_coord]
                send = min(f.remaining, avail)
                if send <= 0:
                    continue

                # Record schedule
                schedules[f.fid].append((t, best_coord[0], best_coord[1], send))

                # Update states
                f.remaining -= send
                caps[best_coord] = avail - send
                if best_coord not in f.landings_used:
                    f.landings_used.add(best_coord)
                    f.unique_landings = len(f.landings_used)
                if f.last_landing != best_coord:
                    # Landing point changed
                    f.last_landing = best_coord

        return schedules
