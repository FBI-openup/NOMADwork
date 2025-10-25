from __future__ import annotations

import argparse
import os
import random
import math
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
    bias_rect_p: float = 0.0,
    bias_access_p: float = 0.0,
    access_radius: int = 3,
    phase_mode: str = "random",  # random|stripe-x|stripe-y|checker|ring
    sparse_b: float = 0.0,        # fraction of UAVs with B forced to 0
    volume_dist: str = "uniform", # uniform|pareto
    starts_dist: str = "uniform", # uniform|bimodal
    rect_shape: str = "uniform",  # uniform|tall|wide
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

    def phase_at(x:int,y:int)->int:
        if phase_mode == "stripe-x":
            return x % 10
        if phase_mode == "stripe-y":
            return y % 10
        if phase_mode == "checker":
            return (x + 2*y) % 10
        if phase_mode == "ring":
            cx, cy = (M-1)/2.0, (N-1)/2.0
            r = int(abs(x - cx) + abs(y - cy))
            return r % 10
        return random.randint(0, 9)

    for y in range(N):
        for x in range(M):
            base_B = random.randint(B_min, B_max)
            B = max(1, int(round(base_B * boost_factor(x, y))))
            # Apply sparsity (holes)
            if random.random() < max(0.0, min(1.0, sparse_b)):
                B = 1
            phase = phase_at(x, y)
            lines.append(f"{x} {y} {B} {phase}")

    # Flows
    for fid in range(1, FN + 1):
        # Access point: optionally bias near hotspots to create contention
        if hotspot_centers and random.random() < max(0.0, min(1.0, bias_access_p)):
            cx, cy = random.choice(hotspot_centers)
            r = max(0, int(access_radius))
            ax = max(0, min(M - 1, cx + random.randint(-r, r)))
            ay = max(0, min(N - 1, cy + random.randint(-r, r)))
        else:
            ax = random.randint(0, M - 1)
            ay = random.randint(0, N - 1)
        # starts
        if starts_dist == "bimodal" and T > 0:
            if random.random() < 0.7:
                t_start = random.randint(max(0, start_min), max(0, (start_min + start_max)//3))
            else:
                t_start = random.randint(max(0, (2*(start_min + start_max))//3), min(T - 1, start_max))
        else:
            t_start = random.randint(max(0, start_min), min(T - 1, start_max)) if T > 0 else 0
        # volume
        if volume_dist == "pareto":
            # Pareto(alpha=2) scaled into [size_min,size_max]
            u = max(1e-6, min(0.999999, random.random()))
            x_p = (1.0 / (u ** (1/2)))  # inverse CDF (approx), heavy tail
            # normalize
            x_p = (x_p - 1.0) / (10.0 - 1.0)
            q_total = int(round(size_min + x_p * (size_max - size_min)))
            q_total = max(size_min, min(size_max, q_total))
        else:
            q_total = random.randint(size_min, size_max)
        # Landing rectangle: optionally bias around hotspots
        if hotspot_centers and random.random() < max(0.0, min(1.0, bias_rect_p)):
            cx, cy = random.choice(hotspot_centers)
            # Choose width/height within bounds, centered on (cx, cy)
            w = random.randint(max(1, rect_min_w), max(1, rect_max_w))
            h = random.randint(max(1, rect_min_h), max(1, rect_max_h))
            x1 = max(0, cx - w // 2)
            y1 = max(0, cy - h // 2)
            x2 = min(M - 1, x1 + w - 1)
            y2 = min(N - 1, y1 + h - 1)
            # Adjust start if rectangle truncated at edges
            x1 = max(0, x2 - w + 1)
            y1 = max(0, y2 - h + 1)
            m1, n1, m2, n2 = x1, y1, x2, y2
        else:
            # shape-biased rectangles
            if rect_shape in ("tall","wide"):
                # pick width/height within bounds but biased
                if rect_shape == "wide":
                    w = random.randint(max(1, rect_min_w + (rect_max_w-rect_min_w)//2), rect_max_w)
                    h = random.randint(rect_min_h, max(rect_min_h, rect_min_h + (rect_max_h-rect_min_h)//3))
                else:
                    h = random.randint(max(1, rect_min_h + (rect_max_h-rect_min_h)//2), rect_max_h)
                    w = random.randint(rect_min_w, max(rect_min_w, rect_min_w + (rect_max_w-rect_min_w)//3))
                x1 = random.randint(0, max(0, M - w))
                y1 = random.randint(0, max(0, N - h))
                x2 = x1 + w - 1
                y2 = y1 + h - 1
                m1, n1, m2, n2 = x1, y1, x2, y2
            else:
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
    ap.add_argument("--B-min", type=int, default=20, help="Min peak bandwidth B")
    ap.add_argument("--B-max", type=int, default=100, help="Max peak bandwidth B")
    ap.add_argument("--size-min", type=int, default=10, help="Min flow size (Mbits)")
    ap.add_argument("--size-max", type=int, default=100, help="Max flow size (Mbits)")
    ap.add_argument("--start-min", type=int, default=0, help="Earliest start time")
    ap.add_argument("--start-max", type=int, default=140, help="Latest start time")
    ap.add_argument("--rect-min-w", type=int, default=4, help="Min landing rectangle width")
    ap.add_argument("--rect-max-w", type=int, default=20, help="Max landing rectangle width")
    ap.add_argument("--rect-min-h", type=int, default=4, help="Min landing rectangle height")
    ap.add_argument("--rect-max-h", type=int, default=20, help="Max landing rectangle height")
    ap.add_argument("--hotspots", type=int, default=0, help="Number of bandwidth hotspots")
    ap.add_argument("--hotspot-boost", type=float, default=1, help="Hotspot B multiplier at center")
    ap.add_argument("--bias-rect-p", type=float, default=0.0, help="Probability to center rectangles near hotspots")
    ap.add_argument("--bias-access-p", type=float, default=0.0, help="Probability to choose access near hotspots")
    ap.add_argument("--access-radius", type=int, default=3, help="Max offset from hotspot for access bias")
    ap.add_argument("--phase-mode", type=str, default="random", help="Phase layout: random|stripe-x|stripe-y|checker|ring")
    ap.add_argument("--sparse-b", type=float, default=0.0, help="Fraction of UAVs with near-zero B to create holes")
    ap.add_argument("--volume-dist", type=str, default="uniform", help="Volume distribution: uniform|pareto")
    ap.add_argument("--starts-dist", type=str, default="uniform", help="Starts distribution: uniform|bimodal")
    ap.add_argument("--rect-shape", type=str, default="uniform", help="Landing rectangle shape: uniform|tall|wide")

    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    # Ensure rectangle bounds don't exceed grid
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
        bias_rect_p=args.bias_rect_p,
        bias_access_p=args.bias_access_p,
        access_radius=args.access_radius,
        phase_mode=args.phase_mode,
        sparse_b=args.sparse_b,
        volume_dist=args.volume_dist,
        starts_dist=args.starts_dist,
        rect_shape=args.rect_shape,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
