"""Determine independent flow clusters based on overlapping landing zones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .candidates import CandidateIndex


@dataclass(slots=True)
class Cluster:
    flow_ids: List[int]
    landing_cells: List[Tuple[int, int]]


def cluster_flows(index: CandidateIndex) -> List[Cluster]:
    cell_to_flows: Dict[Tuple[int, int], Set[int]] = {}
    for flow_id, flow_candidate in index.flows.items():
        for cell in flow_candidate.time_slots:
            cell_to_flows.setdefault(cell, set()).add(flow_id)

    visited: Set[int] = set()
    clusters: List[Cluster] = []

    for flow_id in index.flows:
        if flow_id in visited:
            continue

        queue = [flow_id]
        pointer = 0
        component_flows: Set[int] = set()
        component_cells: Set[Tuple[int, int]] = set()

        while pointer < len(queue):
            current = queue[pointer]
            pointer += 1
            if current in visited:
                continue
            visited.add(current)
            component_flows.add(current)
            candidate = index.flows[current]
            for cell in candidate.time_slots:
                component_cells.add(cell)
                for neighbor in cell_to_flows.get(cell, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)

        clusters.append(
            Cluster(
                flow_ids=sorted(component_flows),
                landing_cells=sorted(component_cells),
            )
        )

    clusters.sort(key=lambda c: len(c.flow_ids), reverse=True)
    return clusters
