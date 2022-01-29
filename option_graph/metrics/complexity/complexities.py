# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" General complexity. """


from typing import Dict, Tuple, TYPE_CHECKING
from copy import deepcopy

from option_graph.option import Option
from option_graph.metrics.complexity.utils import update_sum_dict

if TYPE_CHECKING:
    from option_graph.node import Node


def general_complexity(
    option: Option,
    used_nodes_all: Dict["Node", Dict["Node", int]],
    saved_complexity,
    kcomplexity,
    previous_used_nodes=None,
) -> Tuple[float]:
    """Compute the general complexity of an Option with used nodes.

    Using the number of time each node is used in its OptionGraph and based on the increase of
    complexity given by 'kcomplexity', and the saved complexity using options given by
    'saved_complexity', we sum the general complexity of an option and the total saved complexity.

    Args:
        option: The Option for which we compute the general complexity.
        used_nodes_all: Dictionary of dictionary of the number of times each nodes was used in the
            past, and thus for each node. Not being in a dictionary is counted as not being used.
        saved_complexity: Callable taking as input (node, n_used, n_previous_used), where node is
            the node concerned, n_used is the number of time this node is used and n_previous_used
            is the number of time this node was used before that, returning the saved complexity
            thanks to using an option.
        kcomplexity: Callable taking as input (node, n_used), where node is the node concerned and
            n_used is the number of time this node is used, returning the accumulated complexity.
        previous_used_nodes: Dictionary of the number of times each nodes was used in the past,
            not being in the dictionary is counted as 0.

    Returns:
        Tuple composed of the general complexity and the total saved complexity.

    """

    previous_used_nodes = previous_used_nodes if previous_used_nodes else {}

    total_complexity = 0
    total_saved_complexity = 0

    for node in used_nodes_all[option]:
        n_used = used_nodes_all[option][node]
        n_previous_used = (
            previous_used_nodes[node] if node in previous_used_nodes else 0
        )

        if isinstance(node, Option):
            try:
                node.graph
                node_complexity, saved_node_complexity = general_complexity(
                    node,
                    used_nodes_all,
                    saved_complexity=saved_complexity,
                    kcomplexity=kcomplexity,
                    previous_used_nodes=deepcopy(previous_used_nodes),
                )
                previous_used_nodes = update_sum_dict(
                    previous_used_nodes, used_nodes_all[node]
                )
                total_complexity += saved_node_complexity * kcomplexity(node, n_used)
                total_saved_complexity += saved_node_complexity * kcomplexity(
                    node, n_used
                )
            except NotImplementedError:
                node_complexity = node.complexity
        else:
            node_complexity = node.complexity

        total_complexity += node_complexity * kcomplexity(node, n_used)

        if isinstance(node, Option):
            total_saved_complexity += node_complexity * saved_complexity(
                node, n_used, n_previous_used
            )

        previous_used_nodes = update_sum_dict(previous_used_nodes, {node: n_used})

    return total_complexity - total_saved_complexity, total_saved_complexity


def learning_complexity(
    option: Option,
    used_nodes_all: Dict["Node", Dict["Node", int]],
    previous_used_nodes=None,
):
    """Compute the learning complexity of an Option with used nodes.

    Using the number of time each node is used in its OptionGraph we compute the learning
    complexity of an option and the total saved complexity.

    Args:
        option: The Option for which we compute the learning complexity.
        used_nodes_all: Dictionary of dictionary of the number of times each nodes was used in the
            past, and thus for each node. Not being in a dictionary is counted as not being used.
        previous_used_nodes: Dictionary of the number of times each nodes was used in the past,
            not being in the dictionary is counted as not being used.

    Returns:
        Tuple composed of the learning complexity and the total saved complexity.

    """
    return general_complexity(
        option=option,
        used_nodes_all=used_nodes_all,
        previous_used_nodes=previous_used_nodes,
        saved_complexity=lambda node, k, p: max(0, min(k, p + k - 1)),
        kcomplexity=lambda node, k: k,
    )
