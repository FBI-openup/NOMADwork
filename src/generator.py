from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Tuple


def rand_rect(M: int, N: int, min_w: int, max_w: int, min_h: int, max_h: int) -> Tuple[int, int, int, int]:
    min_w = max(1, min_w)
    min_h = max(1, min_h)
    max_w = max(min_w, min(max_w, M))
    max_h = max(min_h, min(max_h, N))
    w = random.randint(min_w, max_w)
    h = random.randint(min_h, max_h)
    if w > M:
        w = M
    if h > N:
        h = N
    x1 = random.randint(0, M - w)
    y1 = random.randint(0, N - h)
    x2 = x1 + w - 1
    y2 = y1 + h - 1
    return x1, y1, x2, y2


def generate(
    M: int,
    N: int,
    FN: int,
    T: int,
    B_min: int,
    B_max: int,
    size_min: int,
    size_max: int,
    start_min: int,
    start_max: int,
    rect_min_w: int,
    rect_max_w: int,
    rect_min_h: int,
    rect_max_h: int,
    hotspots: int = 0,
    hotspot_boost: float = 2.0,
):
    # Header
    lines = [f"{M} {N} {FN} {T}"]

    # U2GL entries
    # Optionally create hotspot regions with boosted B
    hotspot_centers = []
    for _ in range(max(0, hotspots)):
        hotspot_centers.append((random.randint(0, M - 1), random.randint(0, N - 1)))

    def boost_factor(x: int, y: int) -> float:
        if not hotspot_centers:
            return 1.0
        # Simple nearest-center boost with decay by Manhattan distance
        dmin = min(abs(x - cx) + abs(y - cy) for cx, cy in hotspot_centers)
        # 1 at far, hotspot_boost at center, decay factor 0.5 per hop
        return 1.0 + (hotspot_boost - 1.0) * (0.5 ** dmin)

    for y in range(N):
        for x in range(M):
            base_B = random.randint(B_min, B_max)
            B = max(1, int(round(base_B * boost_factor(x, y))))
            phase = random.randint(0, 9)
            lines.append(f"{x} {y} {B} {phase}")

    # Flows
    for fid in range(1, FN + 1):
        ax = random.randint(0, M - 1)
        ay = random.randint(0, N - 1)
        t_start = random.randint(max(0, start_min), min(T - 1, start_max)) if T > 0 else 0
        q_total = random.randint(size_min, size_max)
        m1, n1, m2, n2 = rand_rect(M, N, rect_min_w, rect_max_w, rect_min_h, rect_max_h)
        lines.append(f"{fid} {ax} {ay} {t_start} {q_total} {m1} {n1} {m2} {n2}")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate UAV U2GL scheduling problem instances")
    ap.add_argument("-M", type=int, required=True, help="Grid width M")
    ap.add_argument("-N", type=int, required=True, help="Grid height N")
    ap.add_argument("-F", "--flows", type=int, required=True, help="Number of flows")
    ap.add_argument("-T", type=int, required=True, help="Total time horizon")
    ap.add_argument("-o", "--output", help="Output file (default stdout)")
    ap.add_argument("--seed", type=int, help="Random seed")
    ap.add_argument("--B-min", type=int, default=10, help="Min peak bandwidth B")
    ap.add_argument("--B-max", type=int, default=200, help="Max peak bandwidth B")
    ap.add_argument("--size-min", type=int, default=10, help="Min flow size (Mbits)")
    ap.add_argument("--size-max", type=int, default=200, help="Max flow size (Mbits)")
    ap.add_argument("--start-min", type=int, default=0, help="Earliest start time")
    ap.add_argument("--start-max", type=int, default=50, help="Latest start time")
    ap.add_argument("--rect-min-w", type=int, default=1, help="Min landing rectangle width")
    ap.add_argument("--rect-max-w", type=int, default=5, help="Max landing rectangle width")
    ap.add_argument("--rect-min-h", type=int, default=1, help="Min landing rectangle height")
    ap.add_argument("--rect-max-h", type=int, default=5, help="Max landing rectangle height")
    ap.add_argument("--hotspots", type=int, default=0, help="Number of bandwidth hotspots")
    ap.add_argument("--hotspot-boost", type=float, default=2.0, help="Hotspot B multiplier at center")

    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    # Ensure rectangle bounds don’t exceed grid
    rect_max_w = min(args.rect_max_w, args.M)
    rect_max_h = min(args.rect_max_h, args.N)

    text = generate(
        args.M,
        args.N,
        args.flows,
        args.T,
        args.B_min,
        args.B_max,
        args.size_min,
        args.size_max,
        args.start_min,
        args.start_max,
        args.rect_min_w,
        rect_max_w,
        args.rect_min_h,
        rect_max_h,
        hotspots=args.hotspots,
        hotspot_boost=args.hotspot_boost,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()

