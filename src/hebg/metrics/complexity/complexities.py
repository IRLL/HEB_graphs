# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2024 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""General complexity."""

from typing import TYPE_CHECKING, Dict, Tuple

from hebg.behavior import Behavior
from hebg.metrics.complexity.utils import update_sum_dict
from hebg.node import Action

if TYPE_CHECKING:
    from hebg.node import Node


def general_complexity(
    behavior: Behavior,
    used_nodes_all: Dict["Node", Dict["Node", int]],
    scomplexity,
    kcomplexity,
    previous_used_nodes: Dict["Node", int] = None,
) -> Tuple[float]:
    """Compute the general complexity of a Behavior with used nodes.

    Using the number of time each node is used in its HEBGraph and based on the increase of
    complexity given by 'kcomplexity', and the saved complexity using behaviors given by
    'saved_complexity', we sum the general complexity of an behavior and the total saved complexity.

    Args:
        behavior: The Behavior for which we compute the general complexity.
        used_nodes_all: Dictionary of dictionary of the number of times each nodes was used in the
            past, and thus for each node. Not being in a dictionary is counted as not being used.
        scomplexity: Callable taking as input (node, n_used, n_previous_used), where node is
            the node concerned, n_used is the number of time this node is used and n_previous_used
            is the number of time this node was used before that, returning the saved complexity
            thanks to using a behavior.
        kcomplexity: Callable taking as input (node, n_used), where node is the node concerned and
            n_used is the number of time this node is used, returning the accumulated complexity.
        previous_used_nodes: Dictionary of the number of times each nodes was used in the past,
            not being in the dictionary is counted as 0.

    Returns:
        Tuple composed of the general complexity and the total saved complexity.

    """

    previous_used_nodes = previous_used_nodes if previous_used_nodes else {}

    total_complexity = 0
    saved_complexity = 0

    for node, n_used in used_nodes_all[behavior].items():
        n_previous_used = (
            previous_used_nodes[node] if node in previous_used_nodes else 0
        )

        if isinstance(node, Behavior) and node in used_nodes_all:
            node_complexity, saved_node_complexity = general_complexity(
                node,
                used_nodes_all,
                scomplexity=scomplexity,
                kcomplexity=kcomplexity,
                previous_used_nodes=previous_used_nodes.copy(),
            )
            previous_used_nodes = update_sum_dict(
                previous_used_nodes, used_nodes_all[node]
            )
            total_complexity += saved_node_complexity * kcomplexity(node, n_used)
            saved_complexity += saved_node_complexity * kcomplexity(node, n_used)
        else:
            node_complexity = node.complexity

        total_complexity += node_complexity * kcomplexity(node, n_used)

        if isinstance(node, (Behavior, Action)):
            saved_complexity += node_complexity * scomplexity(
                node, n_used, n_previous_used
            )

        previous_used_nodes = update_sum_dict(previous_used_nodes, {node: n_used})

    return total_complexity - saved_complexity, saved_complexity


def learning_complexity(
    behavior: Behavior,
    used_nodes_all: Dict["Node", Dict["Node", int]],
    previous_used_nodes=None,
):
    """Compute the learning complexity of a Behavior with used nodes.

    Using the number of time each node is used in its HEBGraph we compute the learning
    complexity of a behavior and the total saved complexity.

    Args:
        behavior: The Behavior for which we compute the learning complexity.
        used_nodes_all: Dictionary of dictionary of the number of times each nodes was used in the
            past, and thus for each node. Not being in a dictionary is counted as not being used.
        previous_used_nodes: Dictionary of the number of times each nodes was used in the past,
            not being in the dictionary is counted as not being used.

    Returns:
        Tuple composed of the learning complexity and the total saved complexity.

    """
    return general_complexity(
        behavior=behavior,
        used_nodes_all=used_nodes_all,
        previous_used_nodes=previous_used_nodes,
        scomplexity=lambda node, k, p: max(0, min(k, p + k - 1)),
        kcomplexity=lambda node, k: k,
    )
