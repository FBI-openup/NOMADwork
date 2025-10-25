#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_inputs.py

生成符合题目输入格式的随机测试用例。

Usage examples:
  python generate_inputs.py --cases 3 --out-dir "./test_inputs" --seed 42
  python generate_inputs.py --cases 1 --M 5 --N 6 --FN 12 --T 120

输出：每个 case 会写为 case_001.txt, case_002.txt ... 到输出目录。

文件格式（与 prior_solver.py 兼容）：
  第1行: M N FN T
  接下来 M*N 行: x y B phi
  接下来 FN 行: f x y t_start Q_total m1 n1 m2 n2

作者：自动生成脚本
"""

import os
import sys
import argparse
import random
from typing import List, Tuple


def _make_uavs(M: int, N: int, B_min: int, B_max: int, rng: random.Random) -> List[Tuple[int,int,int,int]]:
    """返回 M*N 个 UAV 的 (x,y,B,phi)。
    我们按规则把 UAV 坐标铺满网格：x in [0..M-1], y in [0..N-1]。
    B 为整数（峰值带宽），phi 为 0..9 的相位。
    """
    uavs = []
    for x in range(M):
        for y in range(N):
            B = rng.randint(B_min, B_max)
            phi = rng.randint(0, 9)
            uavs.append((x, y, B, phi))
    return uavs


def _make_flows(FN: int, M: int, N: int, T: int, Q_min: int, Q_max: int, rng: random.Random) -> List[Tuple[int,int,int,int,int,int,int,int,int]]:
    """生成 FN 条流，返回列表 (f,x,y,t_start,Q_total,m1,n1,m2,n2)
    - 入口 (x,y) 随机选自 UAV 网格
    - t_start 在 [0, max(0, T-20)]（如果 T 小于 21，则在 [0, T-1]）
    - Q_total 在 [Q_min, Q_max]
    - landing rectangle 随机生成，保证 m1<=m2, n1<=n2
    """
    flows = []
    max_start = max(0, T-20) if T > 1 else 0
    if T <= 1:
        start_upper = 0
    else:
        start_upper = max(0, T-1)

    for f in range(FN):
        x = rng.randint(0, M-1)
        y = rng.randint(0, N-1)
        if T >= 21:
            t_start = rng.randint(0, max_start)
        else:
            t_start = rng.randint(0, start_upper)
        Q_total = rng.randint(Q_min, Q_max)

        # landing rectangle
        m1 = rng.randint(0, M-1)
        m2 = rng.randint(m1, M-1)
        n1 = rng.randint(0, N-1)
        n2 = rng.randint(n1, N-1)

        flows.append((f, x, y, t_start, Q_total, m1, n1, m2, n2))
    return flows


def validate_case(M:int, N:int, FN:int, T:int, uavs, flows) -> None:
    """简单校验生成的数据，发现问题则 raise AssertionError。"""
    assert 1 < M < 70, "M must be 1 < M < 70"
    assert 1 < N < 70, "N must be 1 < N < 70"
    assert 1 <= FN < 5000, "FN must be 1 <= FN < 5000"
    assert 1 < T < 500, "T must be 1 < T < 500"
    assert len(uavs) == M * N
    for (x,y,B,phi) in uavs:
        assert 0 <= x < M and 0 <= y < N
        assert 0 < B < 1000
        assert 0 <= phi < 10
    assert len(flows) == FN
    for (f,x,y,t_start,Q_total,m1,n1,m2,n2) in flows:
        assert 0 <= x < M and 0 <= y < N
        assert 0 <= m1 <= m2 < M
        assert 0 <= n1 <= n2 < N
        assert 0 <= t_start < T
        assert 1 <= Q_total < 3000


def write_case_file(path: str, M:int, N:int, FN:int, T:int, uavs, flows) -> None:
    """把 case 写成一个文本文件，遵循题目输入格式。"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{M} {N} {FN} {T}\n")
        for (x,y,B,phi) in uavs:
            f.write(f"{x} {y} {B} {phi}\n")
        for (flow_tuple) in flows:
            # flow_tuple = (f,x,y,t_start,Q_total,m1,n1,m2,n2)
            f.write(' '.join(str(v) for v in flow_tuple) + "\n")


def generate_cases(out_dir: str, cases: int, seed: int,
                   M_range: Tuple[int,int], N_range: Tuple[int,int],
                   FN_range: Tuple[int,int], T_range: Tuple[int,int],
                   B_range: Tuple[int,int], Q_range: Tuple[int,int]) -> List[str]:
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    created_files = []
    for i in range(1, cases+1):
        # 随机 M, N, FN, T
        M = rng.randint(M_range[0], M_range[1])
        N = rng.randint(N_range[0], N_range[1])
        FN = rng.randint(FN_range[0], FN_range[1])
        T = rng.randint(T_range[0], T_range[1])

        uavs = _make_uavs(M, N, B_range[0], B_range[1], rng)
        flows = _make_flows(FN, M, N, T, Q_range[0], Q_range[1], rng)

        # 验证
        validate_case(M, N, FN, T, uavs, flows)

        filename = os.path.join(out_dir, f"case_{i:03d}.txt")
        write_case_file(filename, M, N, FN, T, uavs, flows)
        created_files.append(filename)
        print(f"Created {filename} (M={M},N={N},FN={FN},T={T})")
    return created_files


def parse_args():
    p = argparse.ArgumentParser(description="Generate random input cases for NOMADwork solver")
    p.add_argument('--cases', type=int, default=1, help='number of cases to generate')
    p.add_argument('--out-dir', type=str, default='./test_inputs', help='output directory')
    p.add_argument('--seed', type=int, default=None, help='random seed (int)')
    p.add_argument('--M', type=int, help='fixed M (optional)')
    p.add_argument('--N', type=int, help='fixed N (optional)')
    p.add_argument('--FN', type=int, help='fixed FN (optional)')
    p.add_argument('--T', type=int, help='fixed T (optional)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randrange(1<<30)

    # default ranges (sensible for testing)
    M_range = (3, 8)
    N_range = (3, 8)
    FN_range = (5, 20)
    T_range = (50, 200)
    B_range = (50, 500)
    Q_range = (100, 2000)

    # override with fixed params if provided
    if args.M is not None:
        M_range = (args.M, args.M)
    if args.N is not None:
        N_range = (args.N, args.N)
    if args.FN is not None:
        FN_range = (args.FN, args.FN)
    if args.T is not None:
        T_range = (args.T, args.T)

    created = generate_cases(args.out_dir, args.cases, seed,
                             M_range, N_range, FN_range, T_range,
                             B_range, Q_range)
    print('\nSummary:')
    for p in created:
        print('  ' + p)
    print(f'Random seed used: {seed}')

