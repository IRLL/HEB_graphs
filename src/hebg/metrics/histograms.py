# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" HEBGraph used nodes histograms computation. """

from typing import TYPE_CHECKING, Dict, List, Tuple
from warnings import warn

import numpy as np

from hebg.behavior import Behavior
from hebg.metrics.complexity.utils import update_sum_dict
from hebg.node import Action, FeatureCondition

if TYPE_CHECKING:
    from hebg import HEBGraph, Node


def behaviors_histograms(
    behaviors: List["Behavior"],
    default_node_complexity: float = 1.0,
) -> Dict["Behavior", Dict["Node", int]]:
    """Compute the used nodes histograms for a list of Behavior.

    Args:
        behaviors: List of Behavior to compute histograms of.
        default_node_complexity: Default node complexity if Node has no attribute complexity.

    Return:
        Dictionary of dictionaries of the number of use for each used node, for each behavior.
    """
    return behaviors_histograms_and_complexites(behaviors, default_node_complexity)[0]


def behaviors_histograms_and_complexites(
    behaviors: List["Behavior"],
    default_node_complexity: float = 1.0,
) -> Tuple[Dict["Behavior", Dict["Node", int]], Dict["Behavior", float]]:
    """Compute the used nodes histograms for a list of Behavior.

    Args:
        behaviors: List of Behavior to compute histograms of.
        default_node_complexity: Default node complexity if Node has no attribute complexity.

    Return:
        Tuple of two elements:
        - Dictionary of dictionaries of the number of use for each used node, for each behavior.
        - Dictionary computed complexity for each behavior.
    """
    histograms: Dict["Behavior", Dict["Node", int]] = {}
    complexities: Dict["Behavior", float] = {}
    for behavior in behaviors:
        try:
            graph = behavior.graph
        except NotImplementedError:
            warn(
                f"Could not load graph for behavior: {behavior}."
                "Skipping histogram computation."
            )
            continue
        histogram, complexity = hebgraph_histogram_and_complexity(
            graph,
            default_node_complexity=default_node_complexity,
        )
        histograms[behavior] = histogram
        complexities[behavior] = complexity
    return histograms, complexities


def hebgraph_histogram_and_complexity(
    graph: "HEBGraph", default_node_complexity: float = 1.0
) -> Tuple[Dict["Node", int], float]:
    """Compute the used nodes histogram for a Behavior.

    Args:
        behavior: Behavior to compute histogram of.
        default_node_complexity: Default node complexity if Node has no attribute complexity.

    Return:
        Tuple composed of two element:
        - Dictionary of the number of use for each used node in the graph.
        - Computed total complexity of the graph.

    """
    nodes_histograms, nodes_complexities = nodes_histograms_and_complexities(
        graph, default_node_complexity
    )
    root = graph.graph["nodes_by_level"][0][0]  # Assumes a single root
    return nodes_histograms[root], nodes_complexities[root]


def cumulated_hebgraph_histogram(
    graph: "HEBGraph", default_node_complexity: float = 1.0
) -> Dict["Node", int]:
    """Unroll the hebgraph histogram by accumulating sub-behaviors histograms.

    Args:
        graph (HEBGraph): The HEBgraph to compute the cumulated histogram of.
        default_node_complexity (float, optional): Default node complexity. Defaults to 1.0.

    Returns:
        Dict[Node, int]: Cumulated histogram of used nodes with unrolled behaviors.
    """
    histogram, _ = hebgraph_histogram_and_complexity(graph, default_node_complexity)
    histograms = {graph.behavior: histogram}
    sub_behaviors = [
        node
        for node in histogram
        if isinstance(node, Behavior) and node != graph.behavior
    ]
    for i, behavior in enumerate(sub_behaviors):
        if behavior.name in graph.all_behaviors:
            sub_behaviors[i] = graph.all_behaviors[behavior.name]

    histograms.update(behaviors_histograms(sub_behaviors, default_node_complexity))

    done = False
    behavior_iteration = {}  # Stoping condition
    while not done:
        behaviors = [node for node in histogram if isinstance(node, Behavior)]
        behavior_iteration, histogram, histograms = _iterate_cumulation(
            graph=graph,
            behaviors=behaviors,
            histogram=histogram,
            histograms=histograms,
            behavior_iteration=behavior_iteration,
            default_node_complexity=default_node_complexity,
        )

        # Recompute behaviors because some might have been added
        behaviors = [node for node in histogram if isinstance(node, Behavior)]

        # We stop when all behaviors have been iterated on enough.
        if all(behavior in behavior_iteration for behavior in behaviors):
            done = all(
                behavior_iteration[behavior] == histogram[behavior]
                for behavior in behaviors
            )

    return histogram


def _iterate_cumulation(
    graph: "HEBGraph",
    behaviors: List["Behavior"],
    histogram: Dict["Node", int],
    histograms: Dict["Behavior", Dict["Node", int]],
    behavior_iteration: Dict["Behavior", int],
    default_node_complexity: float = 1.0,
):
    """Iterate once on every behavior accumulating histograms.

    Args:
        graph (HEBGraph): Current graph being iterated on.
        behaviors (List[Behavior]): List of behaviors to iterate.
        histogram (Dict[Node, int]): Histogram being accumulated.
        histograms (Dict[Behavior, Dict[Node, int]]): Histograms of knowed behaviors.
        behavior_iteration (Dict[Behavior, int]): Number of iterations already done
            for each behavior.

    Returns:
        Tuple of three updated dictionaries. (behavior_iteration, histogram, histograms).
    """
    for behavior in behaviors:
        already_iterated = (
            behavior_iteration[behavior] if behavior in behavior_iteration else 0
        )
        n_used = max(0, histogram[behavior] - already_iterated)
        if n_used == 0:
            continue
        if behavior not in histograms:
            if behavior.name in graph.all_behaviors:
                behavior = graph.all_behaviors[behavior.name]
            try:
                behavior_graph = behavior.graph
            except NotImplementedError:
                if behavior not in behavior_iteration:
                    behavior_iteration[behavior] = 0
                behavior_iteration[behavior] += 1
                continue
            sub_histogram, _ = hebgraph_histogram_and_complexity(
                behavior_graph, default_node_complexity
            )
            histograms[behavior] = sub_histogram
        for _ in range(n_used):
            if behavior in histograms[behavior]:
                histograms[behavior].pop(behavior)
            histogram = update_sum_dict(histogram, histograms[behavior])
            if behavior not in behavior_iteration:
                behavior_iteration[behavior] = 0
            behavior_iteration[behavior] += 1
    return behavior_iteration, histogram, histograms


def nodes_histograms_and_complexities(
    graph: "HEBGraph",
    default_node_complexity: float = 1.0,
    _behaviors_in_search=None,
):
    """Compute the number of times each node of the graph is present
    while computing complexities to find the least complex path.

    Args:
        graph (HEBGraph): HEBGraph to compute the histogram of.
        default_node_complexity (float, optional): Default node complexity if undefined.
            Defaults to 1.0.
        _behaviors_in_search (List[Behavior], optional): List of behaviors already in reccursive
            pile to avoid infinite cycle. Defaults to None.

    Returns:
        Tuple[Dict[Node, Dict[Node, int]], Dict[Node, float]]: Tuple of two values:
        - Dictionary of subnode used for each node. (nodes_used_nodes)
        - Dictionary of computed smallest path complexities for each node. (complexities)
    """
    nodes_by_level = graph.graph["nodes_by_level"]
    depth = graph.graph["depth"]

    _behaviors_in_search = [] if _behaviors_in_search is None else _behaviors_in_search
    _behaviors_in_search.append(str(graph.behavior))

    complexities = {}
    nodes_used_nodes = {}
    for node in graph.nodes:
        if isinstance(node, (Action, FeatureCondition)):
            complexities[node] = node.complexity
            nodes_used_nodes[node] = {node: 1}

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
                node_only_used_behaviors,
                node_only_complexity,
            ) = _get_node_histogram_complexity(
                node,
                default_node_complexity=default_node_complexity,
                behaviors_in_search=_behaviors_in_search,
            )
            node_used_nodes = update_sum_dict(node_used_nodes, node_only_used_behaviors)
            node_complexity += node_only_complexity

            complexities[node] = node_complexity
            nodes_used_nodes[node] = node_used_nodes
    return nodes_used_nodes, complexities


def _successors_by_index(
    graph: "HEBGraph", node: "Node", complexities: Dict["Node", float]
) -> Tuple[Dict[int, List["Node"]], Dict[int, List[float]]]:
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
        if index not in complexities_by_index:
            complexities_by_index[index] = []
        if index not in succ_by_index:
            succ_by_index[index] = []
        complexities_by_index[index].append(succ_complexity)
        succ_by_index[index].append(succ)
    return succ_by_index, complexities_by_index


def _get_node_histogram_complexity(
    node: "Node", behaviors_in_search=None, default_node_complexity: float = 1.0
) -> Tuple[Dict["Node", int], float]:
    """Compute the used nodes histogram and complexity of a single node.

    Args:
        node: The Node from which we want to compute the complexity.
        behaviors_in_search: List of Behavior already in search to avoid circular search.
        default_node_complexity: Default node complexity if Node has no attribute complexity.

    Return:
        Tuple composed of a dictionary of the number of use for each used Node by the given node
            and the given node complexity.

    """

    if node.type == "behavior":
        if behaviors_in_search is not None and str(node) in behaviors_in_search:
            return {}, np.inf
    if node.type in ("action", "feature_condition", "behavior"):
        try:
            node_complexity = node.complexity
        except AttributeError:
            node_complexity = default_node_complexity
        return {node: 1}, node_complexity
    if node.type == "empty":
        return {}, 0
    raise ValueError(f"Unkowned node type {node.type}")
