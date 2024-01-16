# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>
# pylint: disable=protected-access

""" Deterministic layouts """

import networkx as nx
import numpy as np

from hebg.graph import get_roots


def staircase_layout(graph: nx.DiGraph, center=None):
    """Compute specific default positions for an DiGraph.

    Requires graph to have a 'nodes_by_level' attribute.

    Args:
        graph: A networkx DiGraph with 'nodes_by_level' attribute (see compute_levels).
        center (Optional): Center of the graph layout.

    Returns:
        pos: Positions of each node.

    """

    def place_successors(pos, pos_by_level, node, level) -> int:
        if level not in pos_by_level:
            pos_by_level[level] = pos_by_level[level - 1]
        pos_by_level[level] = max(pos[node][0], pos_by_level[level])
        succs = list(graph.successors(node))
        if len(succs) == 0:
            return 1
        succs_order = np.argsort([graph.edges[node, succ]["index"] for succ in succs])
        for index, succ_id in enumerate(succs_order):
            succ = succs[succ_id]
            if succ in pos:
                continue
            pos[succ] = [pos_by_level[level], -level]
            if index == 0:
                pos[node][0] = max(pos[node][0], pos[succ][0])
                pos_by_level[level - 1] = max(pos_by_level[level - 1], pos[node][0])
            n_succs = place_successors(pos, pos_by_level, succ, level + 1)
            pos_by_level[level] += n_succs
        return len(succs)

    graph, _ = nx.drawing.layout._process_params(graph, center, dim=2)
    pos = {}
    pos_by_level = {0: 0}
    for node in get_roots(graph):
        pos[node] = [pos_by_level[0], 0]
        pos_by_level[0] += place_successors(pos, pos_by_level, node, 1)
    return pos
