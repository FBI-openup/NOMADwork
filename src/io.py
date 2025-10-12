from __future__ import annotations

from typing import Dict, List, Tuple, TextIO

from .model import Flow, Problem, UAVNode, Coord


def _parse_tokens(tokens: List[str]) -> Problem:
    it = iter(tokens)

    M = int(next(it))
    N = int(next(it))
    FN = int(next(it))
    T = int(next(it))

    uavs: Dict[Coord, UAVNode] = {}
    # Next M*N lines: x y B f
    for _ in range(M * N):
        x = int(next(it))
        y = int(next(it))
        B = float(next(it))
        f = int(next(it))
        u = UAVNode(x=x, y=y, B=B, phase=f)
        uavs[(x, y)] = u

    flows: List[Flow] = []
    for _ in range(FN):
        fid = int(next(it))
        ax = int(next(it))
        ay = int(next(it))
        t_start = int(next(it))
        total = float(next(it))
        m1 = int(next(it))
        n1 = int(next(it))
        m2 = int(next(it))
        n2 = int(next(it))
        flows.append(Flow(fid, ax, ay, t_start, total, m1, n1, m2, n2))

    return Problem(M=M, N=N, FN=FN, T=T, uavs=uavs, flows=flows)


def parse_stream(stream: TextIO) -> Problem:
    data = stream.read().strip().split()
    return _parse_tokens(data)


def parse_file(path: str) -> Problem:
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().strip().split()
    return _parse_tokens(data)


def format_output(schedules: Dict[int, List[Tuple[int, int, int, float]]]) -> str:
    # schedules[fid] -> list of (t, x, y, z)
    lines: List[str] = []
    for fid in sorted(schedules.keys()):
        entries = schedules[fid]
        p = len(entries)
        lines.append(f"{fid} {p}")
        for t, x, y, z in entries:
            # z as a float; keep reasonable precision
            lines.append(f"{t} {x} {y} {z:.6f}")
    return "\n".join(lines) + "\n"
