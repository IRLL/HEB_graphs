# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>
# pylint: disable=protected-access

""" Metaheuristics based layouts """

from copy import deepcopy

import numpy as np
import networkx as nx

from option_graph.layouts.metaheuristics import simulated_annealing


def leveled_layout_energy(
    graph: nx.DiGraph, center=None, metaheuristic=simulated_annealing
):
    """Compute positions for a leveled DiGraph using a metaheuristic to minimize energy.

    Requires each node to have a 'level' attribute.

    Args:
        graph: A networkx DiGraph.
        center (Optional): Center of the graph layout.

    Returns:
        pos: Positions of each node.
        nodes_by_level: List of nodes by levels.

    """
    graph, center = nx.drawing.layout._process_params(graph, center, dim=2)

    nodes_by_level = graph.graph["nodes_by_level"]
    pos = {}
    step_size = 1 / max(len(nodes_by_level[level]) for level in nodes_by_level)
    spacing = np.arange(0, 1, step=step_size)
    for level in nodes_by_level:
        n_nodes_in_level = len(nodes_by_level[level])
        if n_nodes_in_level > 1:
            positions = np.linspace(
                0, len(spacing) - 1, n_nodes_in_level, endpoint=True, dtype=np.int32
            )
            positions = spacing[positions]
        else:
            positions = [spacing[(len(spacing) - 1) // 2]]

        for i, node in enumerate(nodes_by_level[level]):
            pos[node] = [level, positions[i]]

    def energy(pos, nodes_strenght=1, edges_strenght=2):
        def dist(x, y):
            x_arr, y_arr = np.array(x), np.array(y)
            return np.linalg.norm(x_arr - y_arr)

        energy = 0
        for level in nodes_by_level:
            for node in nodes_by_level[level]:
                energy += nodes_strenght * sum(
                    np.square(dist(pos[node], pos[n]))
                    for n in nodes_by_level[level]
                    if n != node
                )
                energy -= sum(
                    edges_strenght
                    / abs(
                        max(1, graph.nodes[node]["level"] - graph.nodes[pred]["level"])
                    )
                    / max(1e-6, dist(pos[node], pos[pred]))
                    for pred in graph.predecessors(node)
                )
                energy -= sum(
                    edges_strenght
                    / abs(
                        max(1, graph.nodes[node]["level"] - graph.nodes[succ]["level"])
                    )
                    / max(1e-6, dist(pos[node], pos[succ]))
                    for succ in graph.successors(node)
                )

        return energy

    def neighbor(pos):
        pos_copy = deepcopy(pos)
        choosen_node = np.random.choice(list(pos_copy.keys()))
        choosen_level = graph.nodes(data="level")[choosen_node]
        new_pos = [pos_copy[choosen_node][0], np.random.choice(spacing)]
        for n in nodes_by_level[choosen_level]:
            if n != choosen_node and np.all(np.isclose(new_pos, pos_copy[n])):
                pos_copy[choosen_node], pos_copy[n] = (
                    pos_copy[n],
                    pos_copy[choosen_node],
                )
                return pos_copy
        pos_copy[choosen_node] = new_pos
        return pos_copy

    return metaheuristic(pos, energy, neighbor)
