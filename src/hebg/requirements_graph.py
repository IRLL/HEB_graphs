# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>
# pylint: disable=arguments-differ

""" Module for building underlying requirement graphs based on a set of behaviors. """

from __future__ import annotations

from copy import deepcopy
from typing import List

from networkx import DiGraph, descendants

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

    for behavior, graph in zip(behaviors, heb_graphs):
        requirement_degree[behavior] = {}
        for node in graph.nodes():
            if isinstance(node, EmptyNode):
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
                        try:
                            requirement_degree[behavior][following_behavior] -= 1
                        except KeyError:
                            requirement_degree[behavior][following_behavior] = -1

    for behavior, graph in zip(behaviors, heb_graphs):
        for node in graph.nodes():
            if isinstance(node, Behavior):
                try:
                    requirement_degree[behavior][node] += 1
                except KeyError:
                    requirement_degree[behavior][node] = 1

    index = 0
    for behavior, graph in zip(behaviors, heb_graphs):
        for node in graph.nodes():
            if isinstance(node, Behavior) and requirement_degree[behavior][node] > 0:
                if node not in requirements_graph.nodes():
                    requirements_graph.add_node(node)
                index = len(list(requirements_graph.successors(node))) + 1
                requirements_graph.add_edge(node, behavior, index=index)

    for edge in requirements_graph.edges():
        print(edge)
    compute_levels(requirements_graph)
    return requirements_graph
