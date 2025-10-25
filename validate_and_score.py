#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_and_score.py

验证 solver 的输出并按题目给定/合理化的评分公式计算分数。

Usage:
  python validate_and_score.py --input <input_path> --output <output_path> [--alpha 0.1]

输出：打印 per-flow 检查、得分和 case 总分，返回 JSON summary on stdout.

假设（说明）:
- Delay score: T_Delay = 10 / (t_delay + 10), where t_delay = first_tx_time - t_start (if no tx, t_delay = large -> score 0)
- Distance score: weighted average of distance_score(h) where weights are transmitted z amounts
- Landing score: 1 / k where k is number of distinct landing UAVs used (k=0 -> 0)

这些是合理化的假设并与 prior_solver 中的 delay/距离项一致。
"""

import argparse
import json
import math
from collections import defaultdict


def b_value(B, phi, t):
    tt = (phi + t) % 10
    if tt in (0,1,8,9):
        return 0.0
    elif tt in (2,7):
        return B * 0.5
    else:
        return float(B)


def distance_score(h, alpha=0.1):
    return pow(2.0, -alpha * float(h))


def parse_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read().strip().split()
    it = iter(data)
    M = int(next(it)); N = int(next(it)); FN = int(next(it)); T = int(next(it))
    uavs = []
    for _ in range(M*N):
        x = int(next(it)); y = int(next(it)); B = float(next(it)); phi = int(next(it))
        uavs.append({'x':x,'y':y,'B':B,'phi':phi})
    flows = []
    for _ in range(FN):
        f = int(next(it)); x = int(next(it)); y = int(next(it)); t_start = int(next(it)); Q_total = float(next(it));
        m1 = int(next(it)); n1 = int(next(it)); m2 = int(next(it)); n2 = int(next(it))
        flows.append({'f':f,'x':x,'y':y,'t_start':t_start,'Q_total':Q_total,'rect':(m1,n1,m2,n2)})
    return M,N,FN,T,uavs,flows


def parse_output(path):
    # output format: for each flow: line "f p" then p lines: t x y z
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    it = iter(lines)
    results = {}
    while True:
        try:
            head = next(it)
        except StopIteration:
            break
        parts = head.split()
        if len(parts) < 2:
            raise ValueError('Bad output head line: ' + head)
        f = int(parts[0]); p = int(parts[1])
        recs = []
        for _ in range(p):
            ln = next(it)
            t_s,x_s,y_s,z_s = ln.split()
            recs.append({'t':int(t_s),'x':int(x_s),'y':int(y_s),'z':float(z_s)})
        results[f] = recs
    return results


def validate_and_score(input_path, output_path, alpha=0.1, eps=1e-6):
    M,N,FN,T,uavs,flows = parse_input(input_path)
    out = parse_output(output_path)

    # build mapping from coord to uav index and to B,phi
    coord2uav = {(u['x'],u['y']): u for u in uavs}

    total_Q = 0.0
    sum_weighted_score = 0.0
    per_flow_summary = []
    ok = True
    # For capacity checks: map (t, (x,y)) -> allocated sum
    alloc = defaultdict(float)

    for fl in flows:
        f = fl['f']
        Q_total = fl['Q_total']
        total_Q += Q_total
        recs = out.get(f, [])
        transferred = sum(r['z'] for r in recs)
        if transferred > Q_total + eps:
            ok = False
        # per-time uav allocation checks
        for r in recs:
            alloc[(r['t'], (r['x'], r['y']))] += r['z']

        # delay: first tx time - t_start
        if recs:
            first_t = min(r['t'] for r in recs)
            t_delay = max(0, first_t - fl['t_start'])
            T_delay = 10.0 / (t_delay + 10.0)
        else:
            t_delay = None
            T_delay = 0.0

        # distance score: weighted by z
        if transferred > eps:
            weighted = 0.0
            for r in recs:
                src_x, src_y = fl['x'], fl['y']
                h = abs(src_x - r['x']) + abs(src_y - r['y'])
                weighted += r['z'] * distance_score(h, alpha)
            T_dist = weighted / transferred
        else:
            T_dist = 0.0

        # landing points k distinct
        k = len({(r['x'],r['y']) for r in recs})
        T_land = 1.0 / k if k>0 else 0.0

        T_u2g = min(1.0, transferred / Q_total) if Q_total>0 else 0.0

        S_f = 100.0 * (0.4 * T_u2g + 0.2 * T_delay + 0.3 * T_dist + 0.1 * T_land)
        sum_weighted_score += Q_total * S_f

        per_flow_summary.append({
            'f': f,
            'Q_total': Q_total,
            'transferred': transferred,
            'T_u2g': T_u2g,
            't_delay': t_delay,
            'T_delay': T_delay,
            'T_dist': T_dist,
            'k': k,
            'T_land': T_land,
            'S_f': S_f
        })

    # capacity validation
    capacity_violations = []
    for (t,coord), qty in alloc.items():
        if coord not in coord2uav:
            capacity_violations.append((t, coord, qty, 'unknown_uav'))
            ok = False
            continue
        u = coord2uav[coord]
        cap = b_value(u['B'], u['phi'], t)
        if qty > cap + 1e-6:
            capacity_violations.append((t, coord, qty, cap))
            ok = False

    S_total = (sum_weighted_score / total_Q) if total_Q>0 else 0.0

    summary = {
        'input': input_path,
        'output': output_path,
        'ok': ok,
        'S_total': S_total,
        'per_flow': per_flow_summary,
        'capacity_violations': capacity_violations
    }
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--alpha', type=float, default=0.1)
    args = p.parse_args()
    summary = validate_and_score(args.input, args.output, alpha=args.alpha)
    print(json.dumps(summary, indent=2))

