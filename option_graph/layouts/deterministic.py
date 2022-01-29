# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>
# pylint: disable=protected-access

""" Deterministic layouts """

import networkx as nx


def option_graph_default_layout(graph: nx.DiGraph, center=None):
    """Compute specific default positions for an DiGraph.

    Requires graph to have a 'nodes_by_level' attribute.

    Args:
        graph: A networkx DiGraph with 'nodes_by_level' attribute (see compute_levels).
        center (Optional): Center of the graph layout.

    Returns:
        pos: Positions of each node.

    """
    graph, _ = nx.drawing.layout._process_params(graph, center, dim=2)
    nodes_by_level = graph.graph["nodes_by_level"]
    pos = {}
    levels = list(nodes_by_level.keys())
    levels.sort()
    for level in levels:
        for i, node in enumerate(nodes_by_level[level]):
            preds = list(graph.predecessors(node))
            if len(preds) == 0:
                x_pos = i
            elif len(preds) == 1 and graph.edges[preds[0], node]["color"] == "red":
                x_pos = pos[preds[0]][0]
            else:
                other_nodes_x = [pos[n][0] for n in nodes_by_level[level] if n in pos]
                x_pos = 1 + max(other_nodes_x + [0])
            pos[node] = [x_pos, -level]
    return pos
