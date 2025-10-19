# Conversation Summary

- Clarified the "no flow splitting" rule: a flow may transmit through only one landing UAV per time slot but can move to a different landing UAV in later slots if capacity or adjacency makes it preferable.
- Confirmed that waiting after the earliest possible arrival is allowed; the challenge scoring already penalizes unnecessary delay, so the solver can defer transmission to exploit better capacity windows.
- Agreed to model ground capacity directly from the 10-second `b(I+ + t)` pattern and to treat Manhattan distance as the travel-time proxy for reaching any landing cell inside the allowed rectangle.
- Adopted a landing-focused time-expanded formulation: decision variables track Mbps per `(flow, landing cell, time)` and a binary usage flag per time step, effectively compressing the graph to hotspot landing zones while leaving air-path nodes implicit.
- Updated the Julia migration plan to include binaries `u[f, z, t]`, optional `y[f, z]`, landing-churn accounting, sequential relocation handling, and a compact data representation for the landing-zone time expansion.
