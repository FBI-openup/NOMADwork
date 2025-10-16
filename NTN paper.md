This paper presents an approach to enhance resource utilization in Non-terrestrial Networks (NTNs) using a temporal graph-based deterministic routing strategy. The goal is to address the challenges of dynamic network topology and resources when establishing deterministic routing.

# Enhancing Resource Utilization of Non-terrestrial Networks Using Temporal Graph-based Deterministic Routing

## Abstract Summary

[cite_start]Deterministic routing is a promising technology for future **Non-terrestrial Networks (NTNs)** to improve service performance and resource utilization[cite: 6]. [cite_start]However, the dynamic nature of NTN topology and resources makes establishing deterministic routing challenging, particularly in jointly scheduling transmission links and cycles and maintaining stable end-to-end (E2E) paths[cite: 7, 8].

[cite_start]The proposed work introduces an efficient **temporal graph-based deterministic routing strategy**[cite: 9]:
* [cite_start]A **time-expanded graph (TEG)** is used to represent heterogeneous NTN resources in a time-slotted manner[cite: 10].
* [cite_start]The deterministic routing problem is formulated with constraints and transformed into a solvable **Integer Linear Programming (ILP)** problem, which provides a robust but time-consuming performance upper bound[cite: 11, 12].
* [cite_start]To reduce complexity, the TEG is extended into an **Extended TEG (ETEG)** by introducing virtual nodes and edges for uniform resource and traffic representation[cite: 13, 14].
* [cite_start]A **polynomial-time complexity algorithm** is proposed using the ETEG to dynamically select optimal transmission links and cycles on a hop-by-hop basis[cite: 15].
* [cite_start]Simulation results validate that the proposed algorithm, named **DetR**, significantly improves traffic acceptance compared to existing strategies (SPR, STR, CGR) and has a substantially lower running time than the ILP-based strategy (ILPS)[cite: 16, 86, 448, 449].

---

## I. Introduction

[cite_start]NTNs, such as mega-constellations like Starlink and OneWeb, are emerging solutions for global high-speed Internet due to their extensive coverage and bandwidth[cite: 19, 21]. [cite_start]Efficient routing is a key enabler for enhancing service performance and resource utilization in NTNs, despite the challenge posed by their dynamic topology and resources[cite: 30, 31].

Previous routing strategies have limitations:
* [cite_start]**Shortest Path Routing (SPR):** Models the network as a static graph, identifying the minimum delay or hop E2E path, but lacks adaptability to changing conditions[cite: 33, 34, 35].
* [cite_start]**Snapshot Graph-based Routing (STR):** Extends SPR by using a series of static snapshots but may fail to find feasible paths when resources are scarce in a single snapshot[cite: 36, 37].
* [cite_start]**Contact Graph Routing (CGR):** Incorporates caching-and-forwarding but prioritizes the earliest connected E2E routing, potentially compromising optimal delay[cite: 38, 39].
* [cite_start]These strategies determine routing based on bandwidth over a long duration, which can lead to **micro-bursts**, uncertain delays, and congestion[cite: 40, 41].

[cite_start]**Deterministic routing** is promising for NTNs as it facilitates precise scheduling of transmission links and cycles at each hop, ensuring strict adherence to E2E delay and jitter requirements, and enabling dynamic resource allocation[cite: 42, 43, 44]. [cite_start]Challenges for deterministic routing implementation in NTNs include[cite: 46]:
1.  The high complexity of solving **Integer Linear Programming (ILP)** for joint routing and scheduling, failing to meet real-time requirements.
2.  The dynamic nature of network topology and resources complicates the identification of stable E2E routing paths.

---

## II. System Model and Problem Formulation

### A. System Model

[cite_start]The system considers an NTN with $N$ satellites, $V=\{u_1, u_2, \dots, u_N\}$[cite: 88]. [cite_start]The time window is divided into consecutive cycles $\{\tau_h | h \ge 1\}$ of equal duration $|\tau|$[cite: 90]. [cite_start]A **Time-Critical (TC) traffic demand $f$** has a period $T_f$, per-period size $A_f$, injection time $t_f$, and an E2E delay upper bound $B_f$[cite: 90]. [cite_start]The problem is analyzed over a planning horizon $D$ spanning from start-cycle $\tau_{\tilde{h}}$ to end-cycle $\tau_{\hat{h}}$[cite: 91].

[cite_start]A **Time-Expanded Graph (TEG)**, $\mathcal{G}=\{\mathcal{V}, \mathcal{E}, \mathcal{C}, \mathcal{L}\}$, is used to model the heterogeneous network resources in a time-slotted manner[cite: 91]:
* [cite_start]**Nodes $\mathcal{V}$**: $\mathcal{V}=\{u^h | u \in V, \tilde{h} \le h \le \hat{h}\}$, where $u^h$ is satellite $u$ in cycle $\tau_h$[cite: 99, 100].
* [cite_start]**Edges $\mathcal{E}$**: Includes **transmission edges** $\mathcal{E}_t$ (transmission from $u$ to $v$ in cycle $\tau_h$) and **storage edges** $\mathcal{E}_s$ (caching data at $u$ from $\tau_h$ to $\tau_{h+1}$)[cite: 93, 94].
* [cite_start]**Capacity $\mathcal{C}$**: Includes **link capacity** $\mathcal{C}_t$ (max data on transmission edge, in Mb) and **node storage** $\mathcal{C}_s$ (on-board storage resource, in Mb)[cite: 95].
* [cite_start]**Delay $\mathcal{L}$**: Includes **link delay** $\mathcal{L}_t$ (propagation delay, in ms) and **storage delay** $\mathcal{L}_s$ (cross-cycle caching delay, in ms), where $l_{u^{h}, u^{h+1}}=|\tau|$ for $\mathcal{L}_s$[cite: 96].

### B. Constraint Establishment

Binary-valued variables $X=\{x_{u,v}^{h,k} | (u^h, v^k) [cite_start]\in \mathcal{E}\}$ are defined to indicate if $f$ is transmitted or cached along an edge[cite: 101, 104, 105].

The problem formulation includes several constraints:
1.  [cite_start]**E2E transmission constraint (1):** $f$ must be transmitted once from the source $s$ and received once at the destination $d$ within the planning horizon $D$[cite: 106, 107].
2.  [cite_start]**Lossless forwarding constraint (2):** Any intermediate satellite $v$ must forward the received $f$[cite: 116, 117].
3.  [cite_start]**Capacity constraint (3):** The link capacity must be no less than the traffic size $A_f$ for any selected transmission edge[cite: 119, 120].
4.  [cite_start]**Storage constraint (4):** The node storage must be no less than $A_f$ for any selected storage edge[cite: 122, 123].
5.  [cite_start]**Cross-cycle propagation and caching constraints (5a):** Ensure the arrival time of $f$ at $v$ is after the transmission cycle of $u$ but no later than the transmission cycle of $v$[cite: 125, 126].
6.  [cite_start]**Transmission timing constraint (6):** The time $u$ sends $f$ should fall within its transmission cycle[cite: 112].
7.  [cite_start]**Caching timing constraint (7):** Feasible cycles for caching at $v$ should be no earlier than the cycle $f$ arrives at $v$ but earlier than the transmission cycle of $v$[cite: 114, 115].

### C. Problem Formulation

[cite_start]The deterministic routing problem, $P_1$, is formulated to **minimize the E2E delay** (total propagation and caching delay) of $f$ from $s$ to $d$, subject to constraints (1)-(4) and (5)-(7)[cite: 136, 137, 138].

$$P_1: \min_{X} \sum_{(u^h, v^h) \in \mathcal{E}_t, h=\tilde{h}}^{\hat{h}} x_{u,v}^{h,h} \cdot l_{u^h, v^h} + \sum_{(u^h, u^{h+1}) \in \mathcal{E}_s, h=\tilde{h}}^{\hat{h}-1} x_{u,u}^{h,h+1} \cdot l_{u^h, u^{h+1}}$$
$$\text{s.t.} \quad (1)-(4), (5), (6), (7). [cite_start]\quad \text{[cite: 136, 137]}$$

[cite_start]$P_1$ is a nonlinear problem and thus unsolvable with existing ILP solvers due to the term $y_{u,v}\cdot[L_t(u)+L_s(u)+t_f]$ in constraints (5a), (6), and (7), which involves variable-dependent summation and multivariable multiplication[cite: 140, 141].

[cite_start]The paper linearizes these nonlinear constraints by introducing auxiliary binary-valued variables, transforming the variable-dependent summation (Equations (9)-(12)) and multivariable multiplication (Equations (13)-(21)) into linear forms[cite: 141, 142, 153, 154].

[cite_start]The resulting equivalent **ILP problem $P_2$** is solvable by ILP solvers and provides a robust performance upper bound, but its computation is excessively time-consuming for real-time processing[cite: 210, 214, 215].

$$P_2: \min_{X'} \sum_{(u^h, v^h) \in \mathcal{E}_t, h=\tilde{h}}^{\hat{h}} x_{u,v}^{h,h} \cdot l_{u^h, v^h} + \sum_{(u^h, u^{h+1}) \in \mathcal{E}_s, h=\tilde{h}}^{\hat{h}-1} x_{u,u}^{h,h+1} \cdot l_{u^h, u^{h+1}}$$
$$\text{s.t.} \quad (1)-(4), (9), (10), (16)-(24). [cite_start]\quad \text{[cite: 211, 212]}$$

---

## III. Temporal Graph-based Deterministic Routing

### A. Extended Time-Expanded Graph Model

[cite_start]To address the time-efficiency challenge, the original TEG $\mathcal{G}$ is enhanced into an **Extended Time-Expanded Graph (ETEG)**, $\mathcal{G}'=\{\mathcal{V}', \mathcal{E}', \mathcal{C}', \mathcal{L}'\}$, to uniformly represent heterogeneous network resources and traffic transmission requirements[cite: 216, 219].

[cite_start]The construction involves introducing virtual elements (illustrated in Fig. 2 [cite: 204]):
1.  [cite_start]Initialize $\mathcal{V}'$, $\mathcal{E}'$, $\mathcal{C}'$, and $\mathcal{L}'$ with the sets from the original TEG $\mathcal{G}$[cite: 221].
2.  [cite_start]**Virtual source $s'$**: Introduced to $\mathcal{V}'$, with a **virtual transmission edge** $(s', s^{\tilde{h}})$ added to $\mathcal{E}'$, signifying the earliest departure cycle from source $s$[cite: 222].
3.  **Virtual destination $d'$**: Introduced to $\mathcal{V}'$, with **virtual aggregation edges** $\{(d^h, d') | \tilde{h} \le h \le \hat{h}\}$ added to $\mathcal{E}'$. [cite_start]This reduces complexity by designating $d'$ as the unique destination, avoiding traversing potential routing to all $d^h$[cite: 213].
4.  [cite_start]**Capacity and Delay Metrics**: Virtual transmission and aggregation edges are assigned a capacity of $A_f$ and a delay of $0$, as they lack physical counterparts, thus not affecting the E2E delay[cite: 218].

### B. ETEG-based Deterministic Routing Algorithm (DetR)

[cite_start]The deterministic routing problem is equivalent to a **path-finding** problem in the ETEG, rather than solving the ILP $P_2$[cite: 224]. [cite_start]The proposed **ETEG-based deterministic routing algorithm (DetR)** (Algorithm 1) jointly utilizes link capacity and node storage to minimize E2E delay while meeting resource requirements[cite: 225, 226, 227].

[cite_start]The path-finding process establishes a **time-featured path $\mathcal{P}_f$** (Definition 1)[cite: 227, 228]. DetR is essentially a modified shortest path algorithm (like Dijkstra's) implemented on the ETEG:
* [cite_start]**Initialization (Step 2)**: Sets the node delay $L(s')=t_f$ and $L(v^k)=+\infty$ for all other nodes, initializing the priority queue $Q$ with all nodes[cite: 236, 275].
* **Iteration (Steps 3-15)**: Extracts the node $u^h$ with the minimum node delay from $Q$. It then checks neighbors $v^k$ for sufficient capacity ($c_{u^h, v^k} \ge A_f$) and updates the neighbor's node delay if a shorter path is found. [cite_start]Crucially, the updated node is $v^r$ where $r$ is the cycle index corresponding to the arrival time, calculated as $r = \lceil \frac{1}{|\tau|} \cdot (L(u^h) + l_{u^h, v^k}) \rceil$ (Step 8), reflecting cross-cycle propagation and caching constraints[cite: 239, 243, 244, 275].
* [cite_start]**Result (Steps 16-22)**: If $L(d') \le t_f + B_f$, the path $\mathcal{P}_f$ is obtained by backtracking; otherwise, no feasible path exists[cite: 262, 276, 277].

### C. Complexity and Optimality Analyses

* [cite_start]**Optimality (Theorem 1)**: The ETEG-based deterministic routing algorithm is proven to calculate a time-featured path with **minimum E2E delay**[cite: 308]. [cite_start]This is based on the argument that the node extracted from the priority queue has its minimum node delay determined, similar to Dijkstra's algorithm[cite: 309, 313].

* [cite_start]**Time Complexity (Theorem 2)**: The time complexity of DetR is $O(|\mathcal{E}'| \cdot \log|\mathcal{V}'|)$, where $|\mathcal{V}'|$ and $|\mathcal{E}'|$ are the number of nodes and edges in the ETEG, respectively[cite: 315]. [cite_start]This complexity is polynomial and derived from the initialization, iteration, and backtracking steps, which is typical for a priority queue-based shortest path algorithm on a graph with non-negative edge weights[cite: 317, 321].

### D. Algorithm Implementation

[cite_start]An implementation framework based on **segment routing** is proposed for DetR[cite: 325]:
1.  [cite_start]**Parameter maintenance**: The **Network Operations Control Center (NOCC)** gathers link status (capacity, delay, storage)[cite: 327, 328].
2.  [cite_start]**Routing decision**: The NOCC constructs the ETEG, determines the optimal deterministic routing, reserves resources, and configures the deterministic forwarding table for the source satellite[cite: 329, 330].
3.  [cite_start]**Routing deployment**: The source satellite encapsulates per-hop transmission link and cycle information into the TC traffic packets' headers, guiding them to the destination[cite: 331, 332].
4.  **Routing evolution**: The NOCC checks the feasibility of the existing routing for upcoming traffic periods. [cite_start]If feasible, a period-size cycle offset is introduced; otherwise, a re-execution of the routing decision is performed[cite: 334, 335, 336].

---

## IV. Simulations

### A. Simulation Setup

* [cite_start]**Constellation**: A partial Starlink constellation (168 satellites in 12 orbits) is simulated using Satellite Toolkit (STK)[cite: 372, 373, 374].
* **Traffic**: TC traffic demands arrive following a Poisson process over 120 seconds. [cite_start]Each demand has a period of $33.33 \text{ ms}$, per-cycle size of $[0.05, 0.6] \text{ Mb}$, and an E2E delay upper bound $B_f = 75 \text{ ms}$[cite: 360, 361, 374].
* [cite_start]**Network Parameters (Table 1)**[cite: 364]:
    * Cycle duration: $5 \text{ ms}$
    * Link capacity: $[5, 12] \text{ Mb}$
    * Node storage: $\ge 1000 \text{ Mb}$ (implying ample storage capacity)
    * Link delay: $[5, 12] \text{ ms}$
* **Metrics**:
    * [cite_start]**Traffic Acceptance ($\alpha$)**: Total size of demands with E2E deterministic transmission guarantees[cite: 367].
    * [cite_start]**Average Running Time ($\beta$)**: Average time to process a single demand[cite: 368].
    * [cite_start]**Average E2E Path Delay ($\gamma$)**: Average delay of routing paths with deterministic guarantees[cite: 369].
* [cite_start]**Comparison**: DetR is compared against ILPS, SPR, STR, and CGR[cite: 366].

### B. Simulation Results

1.  [cite_start]**Traffic Acceptance ($\alpha$ - Fig. 4 [cite: 356])**:
    * [cite_start]DetR is **comparable to the optimal ILPS**[cite: 419].
    * DetR **surpasses CGR, STR, and SPR**. [cite_start]For an arrival rate of $100 \text{ demands/s}$, $\alpha$ is improved by over $50\%$ compared to the lower-performing algorithms[cite: 419, 420].
    * [cite_start]The superior performance of DetR and ILPS is due to their ability to **jointly utilize link capacity and node storage** across different cycles to establish conflict-free routing[cite: 421].
    * SPR performs the worst due to neglecting time-varying characteristics; [cite_start]STR and CGR show improvements but are limited by basing decisions on average bandwidth, leading to micro-bursts[cite: 422, 423, 424].

2.  [cite_start]**Average Running Time ($\beta$ - Fig. 5 [cite: 414])**:
    * [cite_start]The graph-based algorithms (**DetR, CGR, STR, SPR**) have **significantly lower $\beta$ values than ILPS** (which is several orders of magnitude slower, around $10^4 - 10^5 \text{ us}$)[cite: 426].
    * [cite_start]**DetR is the slowest among the four graph-based algorithms**, but the gap is small (not exceeding $80 \text{ microseconds}$)[cite: 427].
    * [cite_start]The increased running time of DetR is justified by its significant enhancement in traffic acceptance[cite: 429].

3.  [cite_start]**Average E2E Path Delay ($\gamma$ - Fig. 6 [cite: 437])**:
    * [cite_start]The $\gamma$ of DetR **aligns closely with that of ILPS**[cite: 433].
    * [cite_start]$\gamma$ for both gradually increases with the arrival rate as the algorithms include necessary cross-cycle propagation and caching to accommodate more demands, but it **remains below the $75 \text{ ms}$ upper bound**[cite: 434, 438, 439].

---

## V. Conclusion

The study addresses the deterministic routing problem in NTNs. [cite_start]It formulates the problem using TEG and transforms it into a solvable but time-consuming ILP ($P_2$)[cite: 443, 444].

[cite_start]The proposed **ETEG-based deterministic routing algorithm (DetR)** offers a polynomial-time complexity solution[cite: 445]. [cite_start]By jointly utilizing link capacity and node storage, DetR facilitates cross-cycle propagation and caching, enabling it to determine optimal transmission links and cycles hop-by-hop[cite: 446, 447].

[cite_start]Simulations confirm that DetR achieves significantly better **traffic acceptance** than SPR, STR, and CGR, justifying its minor increase in complexity, and exhibits a vastly reduced **running time** compared to ILPS[cite: 448, 449].