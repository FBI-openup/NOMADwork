# Time-varying link resource allocation algorithm

---

## Background

The Unmanned Aerial Vehicle (UAV) network is a significant scenario for 6G connectivity, with inter-UAV routing and networking being a critical technology. In a dynamic, time-varying UAV network, localized congestion can lead to service interruptions. Consequently, there is an urgent need for a traffic allocation and scheduling algorithm to alleviate network congestion and enhance the utilization of network-wide bandwidth. The UAV-to-ground links (U2GL) are identified as the primary bandwidth bottleneck for the entire network. Therefore, it is essential to properly distribute each UAV service data flow to a Ground Station to optimize U2GL bandwidth utilization, data flow delay, and other key metrics.

---

## Question Modeling

We model the UAV network as an $M \times N$ grid. Each UAV node's position is defined by coordinates (x, y), with x ranging from 0 to M-1 and y from 0 to N-1. The system includes multiple data flows, each connecting to a specific UAV and landing within a designated rectangular region of UAVs. The total operational time for the system is T seconds, simulated with a discrete granularity of one second.

The goal is to design a network-wide traffic scheduling scheme. Specifically, each data flow needs to be distributed to one U2GL at each time slot.


The diagram above illustrates the network topology. An $M \times N$ grid of UAVs is shown. A data flow originates at an access UAV with coordinates $(x,y)$ and is routed to a landing UAV at $(x',y')$. This landing UAV must be within a specified "data flow destination region," a rectangular area defined by the corner coordinates $(m1, n1)$ and $(m2, n2)$. From the landing UAV, the data is transmitted to a ground station via the U2GL.

### The U2GL Bandwidth Model

The U2GL bandwidth for each UAV is defined by its coordinates, a peak bandwidth B, and a phase φ: {UAV coordinates (x, y), peak bandwidth B, phase φ}.

The U2GL bandwidth is represented by a periodic function $b(t)$, which has a period of 10 seconds.


As shown in the graph, the bandwidth changes based on the time slot (1 second):
* **0 Mbps:** at time slots 0, 1, 8, and 9.
* $B/2$ **Mbps:** at time slots 2 and 7.
* $B$ **Mbps:** from time slots 3 through 6.
From slot 10 is a repetition of slot 0 through 9, same with slot 20, forever.

The peak bandwidth B can differ for each UAV. Additionally, each UAV has a unique phase, φ. Therefore, the effective bandwidth for a UAV at (x, y) at any given time t is calculated as $b(\varphi+t)$.

### The Model of Each Flow

Each flow is defined by the following parameters:
`{Flow f, Access UAV coordinates (x, y), Flow start time t_start, Total traffic volume s, Range of landing UAVs [m1,n1], [m2, n2]}`.

A flow `f` with a total traffic volume of `s` Mbits accesses the network at UAV node (x, y) at time $t_{start}$. The candidate destinations for this flow form a rectangular region of UAVs. For a destination UAV at coordinates $(x', y')$, the landing condition is $m1 \le x' \le m2$ and $n1 \le y' \le n2$.

---

## Question Input and Output

### Input Format

* **Line 1:** Four integers: M, N, FN, T.
    * Mesh topology size is $M \times N$ (where $1 < M < 70$, $1 < N < 70$).
    * FN is the number of flows ($1 \le FN < 5000$).
    * T is the total simulation time ($1 < T < 500$).
* **Lines 2 to (2 + M*N - 1):** Each line contains four items: x, y, B, and φ.
    * These define the U2GL bandwidth $b(\varphi+t)$ for UAV (x, y).
    * B is the peak bandwidth ($0 < B < 1000$), and φ is the phase ($0 \le \varphi < 10$).
* **Lines (2 + M*N) to end:** Each line contains nine integers: f, x, y, $t_{start}$, $Q_{total}$, m1, n1, m2, and n2.
    * This represents flow `f` of size $Q_{total}$ ($1 \le Q_{total} < 3000$) accessing the network at UAV (x, y) at time $t_{start}$, with a landing range of [m1, n1], [m2, n2].

### Output Format

The output is a scheduling scheme for each flow.
* **Line 1:** `f` and `p`, where `f` is the flow ID and `p` is the number of scheduling records for that flow.
* **Lines 2 to (2 + p - 1):** Each line contains four numbers: `t`, `x`, `y`, and `z`.
    * This indicates that at time `t` (in seconds), the traffic rate of flow `f` on the U2GL of UAV (x, y) is `z` Mbps. `z` is a double, while the other variables are integers.

---

## Scoring Function

For a single flow, the score is a weighted average of four components.

**1. Total U2G Traffic Score (Weight: 0.4)**
This measures the completeness of the data transfer.
$$\frac{\Sigma_{i}q_{i}}{Q_{total}}$$
* $q_{i}$ is the traffic size transmitted to the ground at time $t_{i}$.
* $Q_{total}$ is the total traffic volume of the flow.

**2. Traffic Delay Score (Weight: 0.2)**
This penalizes delays in transmission. $T_{max}$ is a constant, set to 10s.
$$\Sigma_{i}\frac{T_{max}}{t_{i}+T_{max}} \times \frac{q_{i}}{Q_{total}}$$

**3. Transmission Distance Score (Weight: 0.3)**
This penalizes longer routes. $\alpha$ is a constant, set to 0.1.
$$\Sigma_{i}\frac{q_{i}}{Q_{total}} \times 2^{-\alpha d_{i}}$$
* $d_{i}$ is the hop count between the access UAV and the landing UAV.

**4. Landing UAV Point Score (Weight: 0.1)**
This penalizes using too many different landing points for a single flow.
$$\frac{1}{k}$$
* `k` is the number of unique landing UAVs used. It starts at 1 and increments each time the landing point changes.

**Final Total Score (using the consistent scores from the example):**
$$\frac{40}{40+20} \times 97.54 + \frac{20}{40+20} \times 93.085 = 96.055$$

The **Total Score** for the solution is the weighted average of all flow scores:
$$\text{Total Score} = \sum_{j} \frac{Q_{total,j}}{\Sigma_{j}Q_{total,j}} \cdot score_{j}$$

---

## Example

### Example Input & Output

| Standard Input | Standard Output |
| :--- | :--- |
| `3 3 2 10` | `1 4` |
| `0 0 10 3` | `0 0 0 10` |
| `1 0 10 3` | `1 0 0 10` |
| `2 0 10 3` | `2 0 0 10` |
| `0 1 10 3` | `3 0 0 10` |
| `1 1 10 3` | `2 2` |
| `2 1 10 3` | `2 0 1 10` |
| `0 2 10 3` | `3 0 2 10` |
| `1 2 10 3` | |
| `2 2 10 3` | |
| `1 0 0 0 40 0 0 2 2`| |
| `2 0 1 2 20 0 0 2 2`| |


### Scoring Calculation

**Flow 1:**
* **Total U2G Traffic Score:** $\frac{40}{40} = 1.0$
* **Traffic Delay Score:** $\frac{10}{0+10} \times \frac{10}{40} + \frac{10}{1+10} \times \frac{10}{40} + \frac{10}{2+10} \times \frac{10}{40} + \frac{10}{3+10} \times \frac{10}{40} = 0.877$
* **Transmission Distance Score:** $\frac{10}{40} \times 2^{0} + \frac{10}{40} \times 2^{0} + \frac{10}{40} \times 2^{0} + \frac{10}{40} \times 2^{0} = 1.0$
* **Landing UAV Point Score:** Flow 1 uses only one landing UAV (0,0), so k=1. The score is $\frac{1}{1} = 1.0$.
* **Total Score (Flow 1):** $100 \times (0.4 \times 1.0 + 0.2 \times 0.877 + 0.3 \times 1.0 + 0.1 \times 1.0) = 97.54$

**Flow 2:**
* **Total U2G Traffic Score:** $\frac{20}{20} = 1.0$
* **Traffic Delay Score:** $\frac{10}{2+10} \times \frac{10}{20} + \frac{10}{3+10} \times \frac{10}{20} = 0.9545$
* **Transmission Distance Score:** $\frac{10}{20} \times 2^{0} + \frac{10}{20} \times 2^{-0.1 \times 1} = 0.9665$ (Note: This assumes 0 and 1 hop distances, not 1 and 2 hops as stated in the markdown's initial text, to match the PDF's result).
* **Landing UAV Point Score:** Flow 2 uses two different landing points, so k=2. The score is $\frac{1}{2} = 0.5$.
* **Total Score (Flow 2):** $100 \times (0.4 \times 1.0 + 0.2 \times 0.9545 + 0.3 \times 0.9665 + 0.1 \times 0.5) = 93.085$.

**Final Total Score (using original document's scores for consistency):**
$$\frac{40}{40+20} \times 97.54 + \frac{20}{40+20} \times 93.085 = 96.055$$