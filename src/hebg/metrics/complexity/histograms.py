# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" HEBGraph used nodes histograms computation. """

from typing import Dict, List, Tuple

import numpy as np

from hebg import HEBGraph, Node, Option
from hebg.metrics.complexity.utils import update_sum_dict


def nodes_histograms(
    options: List[Option], default_node_complexity: float = 1.0
) -> Dict[Option, Dict[Node, int]]:
    """Compute the used nodes histograms for a list of Option.

    Args:
        options: List of Option to compute histograms of.
        default_node_complexity: Default node complexity if Node has no attribute complexity.

    Return:
        Dictionary of dictionaries of the number of use for each used node, for each option.

    """
    return {
        option: nodes_histogram(
            option, default_node_complexity=default_node_complexity
        )[0]
        for option in options
    }


def nodes_histogram(
    option: Option, default_node_complexity: float = 1.0, _options_in_search=None
) -> Tuple[Dict[Node, int], float]:
    """Compute the used nodes histogram for an Option.

    Args:
        option: Option to compute histogram of.
        default_node_complexity: Default node complexity if Node has no attribute complexity.
        _options_in_search: Options already in search to avoid circular search.

    Return:
        Tuple composed of a dictionary of the number of use for each used node and the total
        complexity.

    """

    graph = option.graph
    nodes_by_level = graph.graph["nodes_by_level"]
    depth = graph.graph["depth"]

    _options_in_search = [] if _options_in_search is None else _options_in_search
    _options_in_search.append(str(option))

    complexities = {}
    nodes_used_nodes = {}

    for level in range(depth + 1)[::-1]:
        for node in nodes_by_level[level]:
            node_complexity = 0
            node_used_nodes = {}

            # Best successors accumulated histograms and complexity
            succ_by_index, complexities_by_index = _successors_by_index(
                graph, node, complexities
            )
            for index, values in complexities_by_index.items():
                min_index = np.argmin(values)
                choosen_succ = succ_by_index[index][min_index]
                node_used_nodes = update_sum_dict(
                    node_used_nodes, nodes_used_nodes[choosen_succ]
                )
                node_complexity += values[min_index]

            # Node only histogram and complexity
            (
                node_only_used_options,
                node_only_complexity,
            ) = _get_node_histogram_complexity(
                node,
                default_node_complexity=default_node_complexity,
                options_in_search=_options_in_search,
            )
            node_used_nodes = update_sum_dict(node_used_nodes, node_only_used_options)
            node_complexity += node_only_complexity

            complexities[node] = node_complexity
            nodes_used_nodes[node] = node_used_nodes

    root = nodes_by_level[0][0]
    return nodes_used_nodes[root], complexities[root]


def _successors_by_index(
    graph: HEBGraph, node: Node, complexities: Dict[Node, float]
) -> Tuple[Dict[int, List[Node]], Dict[int, List[float]]]:
    """Group successors and their complexities by index.

    Args:
        graph: The HEBGraph to use.
        node: The Node from which we want to group successors.
        complexities: Dictionary of complexities for each potential successor node.

    Return:
        Tuple composed of a dictionary of successors for each index
            and a dictionary of complexities for each index.

    """
    complexities_by_index = {}
    succ_by_index = {}
    for succ in graph.successors(node):
        succ_complexity = complexities[succ]
        index = int(graph.edges[node, succ]["index"])
        try:
            complexities_by_index[index].append(succ_complexity)
            succ_by_index[index].append(succ)
        except KeyError:
            complexities_by_index[index] = [succ_complexity]
            succ_by_index[index] = [succ]
    return succ_by_index, complexities_by_index


def _get_node_histogram_complexity(
    node: Node, options_in_search=None, default_node_complexity: float = 1.0
) -> Tuple[Dict[Node, int], float]:
    """Compute the used nodes histogram and complexity of a single node.

    Args:
        node: The Node from which we want to compute the complexity.
        options_in_search: Options already in search to avoid circular search.
        default_node_complexity: Default node complexity if Node has no attribute complexity.

    Return:
        Tuple composed of a dictionary of the number of use for each used Node by the given node
            and the given node complexity.

    """

    if node.type == "option":
        if options_in_search is not None and str(node) in options_in_search:
            return {}, np.inf
    if node.type in ("action", "feature_condition", "option"):
        try:
            node_complexity = node.complexity
        except AttributeError:
            node_complexity = default_node_complexity
        return {node: 1}, node_complexity
    if node.type == "empty":
        return {}, 0
    raise ValueError(f"Unkowned node type {node.type}")
