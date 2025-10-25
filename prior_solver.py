#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pior_solver.py  (spawn-in × dynamic priority with strict fairness)

【Spawn-in 逻辑 | Spawn-in Logic】
- 不在落区外仿真移动。为每个 (flow f, 落地 ell) 预计算 hop_min 和 ETA=t_start+hop_min。
- Do NOT simulate movement outside landing zone. Pre-compute hop_min and ETA=t_start+hop_min
  for each (flow f, landing UAV ell).
- 每秒 t，仅当 t >= ETA[f,ell] 时，(f,ell) 才能在 ell 落地分配。
- At each second t, (f,ell) is eligible for allocation at ell only if t >= ETA[f,ell].

【硬约束 | Hard Constraints】
- 同一秒同一 flow 只允许在一个落地 UAV 落地（单出口/秒）。
  Single exit per second: each flow can land at exactly one UAV per second.
- 每个落地 UAV 在第 t 秒的分配总量 ≤ b_ell(phi+ t)。
  Each landing UAV ell has capacity b_ell(phi+t) Mbps at second t.

【平滑动态优先级 | Smooth Dynamic Priority】
1) 窗口平滑/惯性（Inertia）：上一秒在 ell 落地 → bonus
   If flow landed at ell in previous second → inertia bonus
2) ETA感知（ETA-aware）：t+1 在 ell 有 B 或 B/2 → bonus
   If ell has capacity B or B/2 in next second → foresight bonus
3) 完成度奖励（Completion reward）：剩余占比越小 → 小幅加分
   Smaller remaining ratio → slight priority boost (helps wrap up flows)
4) 曼哈顿距离（Manhattan distance）：h = |src_x - uav_x| + |src_y - uav_y|
   Used in distance_score(h) = 2^{-β*h}, penalizes far landing zones

【硬核严格优先级 | Strict Priority with Deterministic Tie-break】
- 移除饥饿保护：完全按优先级分配，无补偿机制
  Remove starvation protection: allocate purely by priority, no fairness compensation
- Tie-break keys (when primary priority is equal):
  a) 更早 ETA (Earlier ETA)：-ETA[(f, ell)]
  b) 更短跳数 (Shorter hop)：-distance_score(h_val)
  c) 更小剩余 (Smaller remaining)：-1.0/remain (easier to wrap up)
  d) 更小 ID (Smaller flow ID)：-f (deterministic, reproducible)

I/O:
  stdin:
    Line 1: M N FN T
    Lines 2..(1+M*N): x y B phi
    Lines (2+M*N)..end: f x y t_start Q_total m1 n1 m2 n2
  stdout (per flow f):
    Line 1: f p
    Next p lines: t x y z   (z in Mbps for second t on UAV(x,y))

Author: Solver
"""

import sys

# ==========================
# Data structures
# ==========================

class UAV:
    __slots__ = ("x","y","B","phi")
    def __init__(self, x:int, y:int, B:float, phi:int):
        self.x = x
        self.y = y
        self.B = float(B)
        self.phi = int(phi)

class Flow:
    __slots__ = ("f","src_pos","t_start","Q_total","remain","land_rect",
                 "active","done","k","last_landing","starve_cnt",
                 "records","last_landed_this_second")
    def __init__(self, f:int, src_xy, t_start:int, Q_total:float, land_rect):
        self.f = f                  # Flow ID (matches problem notation)
        self.src_pos = tuple(src_xy) # Source position (x, y)
        self.t_start = int(t_start) # Flow start time
        self.Q_total = float(Q_total)  # Mbit total (matches problem notation)
        self.remain = float(Q_total)   # Mbit remaining
        self.land_rect = land_rect     # Landing rectangle (m1,n1,m2,n2)
        self.active = False
        self.done = False
        self.k = 0                     # Number of landing points used
        self.last_landing = None       # Last landing UAV index used
        self.starve_cnt = 0
        self.records = []              # (t, x, y, z) scheduling records
        self.last_landed_this_second = False

# ==========================
# Problem helpers
# ==========================

def b_value(uav: UAV, t: int) -> float:
    """U2GL capacity at time t for this UAV, in Mbps."""
    tt = (uav.phi + t) % 10
    if tt in (0,1,8,9):
        return 0.0
    elif tt in (2,7):
        return uav.B * 0.5
    else:  # 3,4,5,6
        return uav.B

def in_rect(x,y,rect):
    (m1,n1,m2,n2) = rect
    return (m1 <= x <= m2) and (n1 <= y <= n2)

# ==========================
# Priority (weights & bonuses)
# ==========================

# 基础权重 | Base weights - OPTIMIZED FOR AVERAGE PERFORMANCE ON LARGE GRIDS
# 题目权重：0.4(流量) + 0.2(延迟) + 0.3(距离) + 0.1(稳定性)
# 目标：追求平均性能最优（500*500网格）
W_TRAFFIC = 40.0             # 流量权重
W_DELAY   = 20.0             # 延迟权重
W_DIST    = 12.0             # 距离权重 - REDUCED from 30.0 to allow more diversification
W_LAND    = 10.0             # 稳定性权重

# 平滑动态优先级增强超参数 | Smooth Dynamic Priority Hyper-parameters
INERTIA_BONUS = 1.2          # 上秒在同一落地 - REDUCED from 2.0 to allow more switching
ETA_BONUS_B   = 2.2          # 下一秒 B 窗口 - TUNED
ETA_BONUS_HB  = 1.1          # 下一秒 B/2 窗口 - TUNED
ETA_PENALTY_ZERO = -2.5    # 下一秒为 0 的惩罚（可调，建议与 ETA_BONUS_B 同量级）
COMPLETE_BONUS_ALPHA = 0.12  # 完成度奖励系数 - INCREASED from 0.05

# 【已禁用】饥饿保护 | 【DISABLED】Starvation Protection
STARVE_THRESH = 10**9        # 永远不会触发 / Never reached
STARVE_BOOST  = 0.0          # 不加成 / No bonus applied

# 其他超参数 | Other Hyper-parameters
SWITCH_EPS_FACTOR = 0.6      # 切换阈值系数 - REDUCED from 1.2 for more flexibility
BETA_HOP = 0.1               # 曼哈顿距离衰减系数 / Manhattan distance decay
TOP_K = 10000                # 保留所有候选（对于500*500网格）| Keep many candidates for large grids

# 【新增】可预见未来窗口（Future-window foresight） | Future Window Hyper-parameters
H_WINDOW = 3                 # 往前看 H 秒的滚动窗口 | Look-ahead window (seconds)
HB_RISING_BONUS = 1.4        # B/2 上升段(τ=2→3)的奖励 | Rising B/2 bonus
HB_FALLING_PENALTY = -1.5    # B/2 下降段(τ=7→0)的惩罚 | Falling B/2 penalty
FEASIBILITY_WINDOW = 10      # 可行性判断的时间窗口(秒) | Feasibility check window (seconds)
FEASIBILITY_FACTOR = 1.0     # 可行性压力系数 | Feasibility pressure factor

def delay_weight(t: int) -> float:
    # 延迟权重函数 | Delay weight function
    # 早期时刻权重更高（鼓励早传输）
    # Earlier time slots get higher weight (encourage early transmission)
    # ≈ 10/(t+10)
    return 10.0 / (t + 10.0)

def distance_score(hops:int) -> float:
    # 曼哈顿距离衰减函数 | Manhattan distance decay function
    # h = |src_x - uav_x| + |src_y - uav_y|
    # 距离越远，得分越低（指数衰减）→ 偏好近距离落地
    # Score decreases exponentially with distance → prefer closer landing zones
    # distance_score(h) = 2^{-β*h}，其中 β=0.1（可调参数）
    # distance_score(h) = 2^{-β*h}, where β=0.1 (tunable)
    return pow(2.0, -BETA_HOP * float(hops))

def landing_switch_penalty(k_current:int) -> float:
    # 落地切换惩罚 | Landing switch penalty
    # 当 flow 从第 k 个落地切换到第 k+1 个时，稳定性项的变化 Δ(1/k)
    # When flow switches from k-th to (k+1)-th landing, stability term changes by Δ(1/k)
    # 返回负值（惩罚）→ 鼓励保持在同一落地
    # Returns negative value (penalty) → encourage staying at same landing UAV
    k = max(1, k_current)
    return (1.0 / (k+1)) - (1.0 / k)

def cap_window(uav: UAV, t: int, H: int) -> float:
    """
    【新增】计算 UAV 在时刻 t 往后 H 秒的总可用容量 | Calculate total available capacity for next H seconds
    cap_window(u, t, H) = Σ_{k=0..H-1} b_value(u, t+k)

    Args:
        uav: UAV object
        t: current time slot
        H: look-ahead window (seconds)
    Returns:
        Total capacity in the next H seconds (Mbps)
    """
    total = 0.0
    for k in range(H):
        total += b_value(uav, t + k)
    return total

def is_tau_rising_hb(uav: UAV, t: int) -> bool:
    """
    【新增】判断当前 τ 是否为 B/2 上升段(τ=2)，下一秒将进入 B 段(τ=3) | Check if currently in rising B/2 phase
    """
    tau_now = (uav.phi + t) % 10
    return tau_now == 2

def is_tau_falling_hb(uav: UAV, t: int) -> bool:
    """
    【新增】判断当前 τ 是否为 B/2 下降段(τ=7)，即将进入 0 段 | Check if currently in falling B/2 phase
    """
    tau_now = (uav.phi + t) % 10
    return tau_now == 7

def feasibility_check(flow, uavs, flow2cands, t: int, T: int) -> float:
    """
    【新增】快速可行性判断：乐观上界估计流 flow 是否能在时间窗口内完成
    Fast feasibility check: optimistic upper bound for flow completion

    Returns:
        feasibility_score ∈ [0, 1]:
          1.0 = 充分(sufficient capacity), 0.5 = 边界(marginal), 0.0 = 紧缺(tight)
    """
    if flow.remain <= 1e-9:
        return 1.0  # Already done

    # 乐观上界：候选落地区域所有UAV的总容量（未来 FEASIBILITY_WINDOW 秒）
    # Optimistic bound: sum of all candidate UAVs' capacity in future window
    future_window_end = min(t + FEASIBILITY_WINDOW, T)
    future_window_len = max(1, future_window_end - t)

    # 计算所有候选落地的总容量上界（简单求和，不考虑竞争）
    total_future_capacity = 0.0
    for ell_idx in flow2cands.get(flow.f, []):
        total_future_capacity += cap_window(uavs[ell_idx], t, future_window_len)

    # 乐观评分：若充足，则 1.0；若紧缺，则更低
    upper_bound = total_future_capacity
    if upper_bound >= flow.remain * 2.0:
        return 1.0  # Highly confident
    elif upper_bound >= flow.remain:
        return 0.75  # Marginal but possible
    elif upper_bound >= flow.remain * 0.5:
        return 0.5  # Tight, need to act
    else:
        return 0.2  # Very tight, critical

# ==========================
# Parsing & precompute
# ==========================

def parse_input(stdin=None):
    """
    Parse input from stdin.
    Input format matches problem specification exactly.
    """
    data = sys.stdin.read().strip().split()
    it = iter(data)
    M = int(next(it)); N = int(next(it)); FN = int(next(it)); T = int(next(it))
    uavs = []
    coord2idx = {}
    for _ in range(M*N):
        x = int(next(it)); y = int(next(it)); B = float(next(it)); phi = int(next(it))
        idx = len(uavs)
        uavs.append(UAV(x, y, B, phi))
        coord2idx[(x, y)] = idx
    flows = []
    for _ in range(FN):
        # Parse flow: f, x, y, t_start, Q_total, m1, n1, m2, n2 (9 integers per problem spec)
        f = int(next(it))
        x = int(next(it))
        y = int(next(it))
        t_start = int(next(it))
        Q_total = int(next(it))
        m1 = int(next(it))
        n1 = int(next(it))
        m2 = int(next(it))
        n2 = int(next(it))
        flows.append(Flow(f, (x, y), t_start, float(Q_total), (m1, n1, m2, n2)))
    flows.sort(key=lambda fl: fl.f)
    return M, N, FN, T, uavs, coord2idx, flows

def pick_topK_landings_for_flow(flow: Flow, uavs, K=TOP_K):
    # Top-K 落地候选选择 | Top-K Landing Candidate Selection
    # 每个 flow 只考虑落地矩形内的前 K 个"最优"UAV
    # Each flow considers at most K "best" UAVs within its landing rectangle
    # 选择标准：0.7*距离分(近优先) + 0.3*信号强度(容量大优先)
    # Selection criteria: 0.7*distance_score(closer preferred) + 0.3*capacity(larger preferred)
    m1, n1, m2, n2 = flow.land_rect
    src_x, src_y = flow.src_pos
    candidates = []
    for idx, u in enumerate(uavs):
        if not in_rect(u.x, u.y, flow.land_rect):
            continue
        # 曼哈顿距离 | Manhattan distance: h = |Δx| + |Δy|
        h = abs(src_x - u.x) + abs(src_y - u.y)
        capacity_signal = u.B
        # 混合启发式：0.7 权重给距离，0.3 权重给容量信号
        # Hybrid heuristic: 0.7 weight for distance, 0.3 weight for capacity signal
        heuristic = 0.7 * distance_score(h) + 0.3 * (capacity_signal / (capacity_signal + 1.0))
        candidates.append((heuristic, idx, h))
    # 按启发式分数降序排列，同分时按跳数升序排列
    # Sort by heuristic score (descending), tie-break by hop count (ascending)
    candidates.sort(key=lambda x: (-x[0], x[2]))
    return [idx for _, idx, _ in candidates[:max(1, K)]]

# ==========================
# Scheduler (spawn-in × priority)
# ==========================

def run_scheduler(M,N,FN,T,uavs,coord2idx,flows):
    # 1) 为每个 flow 选 Top-K 候选落地
    # Step 1: Select Top-K landing candidates for each flow
    flow2cands = {fl.f: pick_topK_landings_for_flow(fl, uavs, TOP_K) for fl in flows}

    # 2) 预计算 hop_min 与 ETA（源→落地）| Precompute hop_min and ETA
    # hop_min[(f, ell)] = 曼哈顿距离 | Manhattan distance from flow source to landing UAV
    # ETA[(f, ell)] = t_start + hop_min：flow 最早可在该落地时刻
    # ETA[(f, ell)] = t_start + hop_min: earliest time flow can land at this UAV
    hop_min = {}  # (f, ell_idx) -> hops
    ETA = {}      # (f, ell_idx) -> earliest time to land
    for fl in flows:
        src_x, src_y = fl.src_pos
        for ell_idx in flow2cands[fl.f]:
            u = uavs[ell_idx]
            h = abs(src_x - u.x) + abs(src_y - u.y)
            hop_min[(fl.f, ell_idx)] = h
            ETA[(fl.f, ell_idx)] = fl.t_start + h

    # 3) 惯性映射：上一秒落地点 | Inertia map: previous second landing point
    last_used_map = {}  # f -> ell_idx

    def capacity_window_flag(ell_idx, t):
        """看 t+1 的容量：2->B, 1->B/2, 0->none"""
        cnext = b_value(uavs[ell_idx], t+1)
        if cnext <= 1e-9: return 0
        if abs(cnext - uavs[ell_idx].B) < 1e-9: return 2
        return 1

    # 初始化 flow 状态
    for fl in flows:
        fl.active = False
        fl.done = False
        fl.k = 0
        fl.last_landing = None
        fl.starve_cnt = 0
        fl.records = []

    # 平滑动态优先级计算 | Smooth Dynamic Priority Computation
    def compute_priority(flow: Flow, ell_idx: int, t: int, cap_flag_next) -> float:
        # 【基础优先级项 | Base Priority Terms】
        # 流量项：1/Q_total（小流高优先级）| Traffic term: 1/Q_total (smaller flows get higher priority)
        base_traffic = W_TRAFFIC * (1.0 / max(flow.Q_total, 1e-9))
        # 延迟项：10/(t+10) * 1/Q_total（早期、小流双重加权）
        # Delay term: 10/(t+10) * 1/Q_total (early time slots and smaller flows emphasized)
        base_delay   = W_DELAY   * (10.0/(t+10.0)) * (1.0 / max(flow.Q_total, 1e-9))
        # 距离项：2^(-β*h)（近距离落地优先）| Distance term: 2^(-β*h) (prefer closer landing)
        h = hop_min[(flow.f, ell_idx)]
        base_dist    = W_DIST    * pow(2.0, -BETA_HOP * float(h))

        # 【稳定性项 | Stability Term】
        # 落地切换惩罚：若本次切换落地点，则扣分
        # Switch penalty: if landing at different UAV from previous, apply negative penalty
        land_term = 0.0
        if (flow.last_landing is not None) and (flow.last_landing != ell_idx):
            land_term = W_LAND * landing_switch_penalty(flow.k)  # negative value

        # 【平滑动态增强项 | Smooth Dynamic Enhancement Bonus】
        bonus = 0.0
        # 1) 惯性加成：如果上秒在同一落地，保持习惯
        # Inertia bonus: if flow landed at this UAV in previous second, encourage continuity
        if last_used_map.get(flow.f) == ell_idx:
            bonus += INERTIA_BONUS

        # 2) 【增强】ETA感知：基于未来 H 秒的滚动窗口容量
        # 【ENHANCED】ETA-aware bonus: based on future H-second rolling window capacity
        u = uavs[ell_idx]
        cap_win = cap_window(u, t, H_WINDOW)
        cap_now = b_value(u, t)

        # 先判断当前是 B/2 的特殊段
        tau_now = (u.phi + t) % 10

        if tau_now == 2:  # 当前 B/2 上升段 | Current is B/2 rising phase
            bonus += HB_RISING_BONUS
        elif tau_now == 7:  # 当前 B/2 下降段 | Current is B/2 falling phase
            bonus += HB_FALLING_PENALTY

        # 再看整个窗口的容量情况
        if cap_win > u.B * H_WINDOW * 0.8:  # 窗口容量充分 | Window has sufficient capacity
            bonus += ETA_BONUS_B
        elif cap_now <= 1e-9:  # 当前为 0 | Zero capacity now
            bonus += ETA_PENALTY_ZERO
        elif cap_now >= u.B * 0.4:  # 当前至少是 B/2 | Current is at least B/2
            bonus += ETA_BONUS_HB

        # 3) 完成度奖励：快要完成的流优先级提高（帮助收尾）
        # Completion bonus: nearly-completed flows get slight boost (helps wrap up)
        remain_ratio = flow.remain / max(flow.Q_total, 1e-9)
        bonus += COMPLETE_BONUS_ALPHA * (1.0 - remain_ratio)

        # 4) 【新增】可行性约束：当可行性低时，提升选择压力
        # 【NEW】Feasibility constraint: when feasibility is low, increase selection pressure
        feas_score = feasibility_check(flow, uavs, flow2cands, t, T)
        if feas_score < 0.3:
            # 紧急情况：优先选择当前有容量的 | Emergency: prefer UAVs with current capacity
            bonus += feas_score * 2.0 * cap_now
        # 【禁用】饥饿保护 → 只按优先级，不做公平补偿
        # 【DISABLED】Starvation protection → allocate purely by priority, no fairness compensation

        return base_traffic + base_delay + base_dist + land_term + bonus

    # 逐秒模拟（仅落地层）| Per-second simulation (landing layer only)
    for t in range(T):
        # 激活新 flow | Activate new flows when they become active
        for fl in flows:
            if (not fl.active) and (t >= fl.t_start) and (not fl.done):
                fl.active = True

        # 本秒每个落地 UAV 的容量 | Current landing capacity for each UAV at time t
        landing_capacity = {ell_idx: b_value(u, t) for ell_idx,u in enumerate(uavs)}
        cap_flag_next = {ell_idx: capacity_window_flag(ell_idx, t) for ell_idx in range(len(uavs))}

        # 本秒是否已为某 flow 分配（单出口/秒）| Track which flows already got allocation this second
        assigned_this_second = set()

        # 【改进】全局优先级排序而不是按UAV遍历
        # 【IMPROVED】Global priority sorting instead of UAV-by-UAV iteration
        # 生成所有可能的(flow, UAV)对，然后全局排序，确保全局最优而不是局部最优

        all_options = []  # (priority_key, flow, ell_idx)

        for fl in flows:
            if not fl.active or fl.done or fl.remain <= 1e-9:
                continue
            if fl.f in assigned_this_second:
                continue

            for ell_idx in flow2cands[fl.f]:
                C = landing_capacity.get(ell_idx, 0.0)
                if C <= 1e-9:
                    continue
                if t < ETA[(fl.f, ell_idx)]:
                    continue

                # 计算该(flow, UAV)对的优先级
                pr = compute_priority(fl, ell_idx, t, cap_flag_next)
                eta_val = ETA[(fl.f, ell_idx)]
                h_val   = hop_min[(fl.f, ell_idx)]
                rem     = fl.remain
                f_val   = fl.f

                key = (pr, -eta_val, -distance_score(h_val), -1.0/rem if rem>0 else 0.0, -f_val)
                all_options.append((key, fl, ell_idx))

        # 全局排序：按优先级从高到低
        # Global sort: highest priority first
        all_options.sort(key=lambda x: x[0], reverse=True)

        # 全局分配：不再考虑UAV本地容量限制，而是全局容量管理
        # Global allocation: respect global capacity constraints
        for key, fl, ell_idx in all_options:
            # 再次检查是否已分配或完成
            if fl.f in assigned_this_second or fl.remain <= 1e-9:
                continue

            # 检查UAV容量
            C = landing_capacity.get(ell_idx, 0.0)
            if C <= 1e-9:
                continue

            # 实际分配
            z = min(fl.remain, C)
            if z <= 1e-9:
                continue

            ux, uy = uavs[ell_idx].x, uavs[ell_idx].y
            fl.records.append((t, ux, uy, z))
            fl.remain -= z
            landing_capacity[ell_idx] -= z
            assigned_this_second.add(fl.f)
            fl.last_landed_this_second = True

            # 更新稳定性计数器
            if fl.last_landing is None:
                fl.last_landing = ell_idx
                fl.k = 1
            elif fl.last_landing != ell_idx:
                fl.k += 1
                fl.last_landing = ell_idx

        # 本秒统计：饥饿/完成/惯性 | Per-second statistics: starvation/completion/inertia
        last_used_map.clear()
        for fl in flows:
            if not fl.active or fl.done:
                continue
            if fl.f in assigned_this_second:
                fl.starve_cnt = 0      # 本秒获得分配 → 重置饥饿计数 | Got allocation → reset starvation counter
                last_used_map[fl.f] = fl.last_landing
            else:
                # 本秒没分到带宽且仍未完成 → 饥饿+1（仅用于日志，不再影响优先级）
                # No allocation this second → increment starvation counter
                # (Counter maintained for logging, but no longer affects priority)
                if t >= fl.t_start and fl.remain > 1e-9:
                    fl.starve_cnt += 1
            if fl.remain <= 1e-9:
                fl.done = True

    # 输出 | Output
    out_lines = []
    for fl in flows:
        p = len(fl.records)
        out_lines.append(f"{fl.f} {p}")
        for (t, x, y, z) in fl.records:
            # 漂亮地输出 z | Pretty-print bandwidth z
            if abs(z - round(z)) < 1e-9:
                z_str = str(int(round(z)))
            else:
                z_str = f"{z:.6f}".rstrip('0').rstrip('.')
            out_lines.append(f"{t} {x} {y} {z_str}")
    return "\n".join(out_lines)

# ==========================
# main
# ==========================

def main():
    M,N,FN,T,uavs,coord2idx,flows = parse_input()
    result = run_scheduler(M,N,FN,T,uavs,coord2idx,flows)
    sys.stdout.write(result + ("\n" if not result.endswith("\n") else ""))

if __name__ == "__main__":
    main()
