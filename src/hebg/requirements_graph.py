# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>
# pylint: disable=arguments-differ

""" Module for building underlying requirement graphs based on a set of behaviors. """

from __future__ import annotations

from copy import deepcopy
from typing import List, Dict

from networkx import DiGraph, descendants

from hebg.heb_graph import HEBGraph
from hebg.graph import compute_levels
from hebg.node import EmptyNode
from hebg.behavior import Behavior


def build_requirement_graph(behaviors: List[Behavior]) -> DiGraph:
    """Builds a DiGraph of the requirements induced by a list of behaviors.

    Args:
        behaviors: List of Behaviors to build the requirement graph from.

    Returns:
        The requirement graph induced by the given list of behaviors.

    """

    try:
        heb_graphs = [behavior.graph for behavior in behaviors]
    except NotImplementedError as error:
        user_msg = "All behaviors given must be able to build an HEBGraph"
        raise NotImplementedError(user_msg) from error

    requirements_graph = DiGraph()
    for behavior in behaviors:
        requirements_graph.add_node(behavior)

    requirement_degree = {}

    for graph in heb_graphs:
        requirement_degree[graph.behavior] = {}
        for node in graph.nodes():
            if not isinstance(node, EmptyNode):
                continue
            requirement_degree = _cut_alternatives_to_empty_node(
                graph, node, requirement_degree
            )

    for graph in heb_graphs:
        for node in graph.nodes():
            if not isinstance(node, Behavior):
                continue
            if node not in requirement_degree[graph.behavior]:
                requirement_degree[graph.behavior][node] = 0
            requirement_degree[graph.behavior][node] += 1

    index = 0
    for graph in heb_graphs:
        for node in graph.nodes():
            if (
                not isinstance(node, Behavior)
                or requirement_degree[graph.behavior][node] == 0
            ):
                continue
            if node not in requirements_graph.nodes():
                requirements_graph.add_node(node)
            index = len(list(requirements_graph.successors(node))) + 1
            requirements_graph.add_edge(node, graph.behavior, index=index)

    compute_levels(requirements_graph)
    return requirements_graph


def _cut_alternatives_to_empty_node(
    graph: HEBGraph,
    node: EmptyNode,
    requirement_degree: Dict[Behavior, Dict[Behavior, int]],
) -> Dict[Behavior, Dict[Behavior, int]]:
    successor = list(graph.successors(node))[0]
    empty_index = graph.edges[node, successor]["index"]
    alternatives = graph.predecessors(successor)
    alternatives = [
        alt_node
        for alt_node in alternatives
        if graph.edges[alt_node, successor]["index"] == empty_index
    ]
    cut_graph = deepcopy(graph)
    for alternative in alternatives:
        cut_graph.remove_edge(alternative, successor)
    for alternative in alternatives:
        following_behaviors = [
            following_node
            for following_node in descendants(cut_graph, alternative)
            if isinstance(following_node, Behavior)
        ]
        for following_behavior in following_behaviors:
            if following_behavior not in requirement_degree[graph.behavior]:
                requirement_degree[graph.behavior][following_behavior] = 0
            requirement_degree[graph.behavior][following_behavior] -= 1

    return requirement_degree
