# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" General complexity. """

from typing import Dict, Tuple
from copy import deepcopy

from option_graph import Option, Node
from option_graph.metrics.complexity.utils import update_sum_dict

def general_complexity(option:Option, used_nodes_all:Dict[Node, Dict[Node, int]],
        saved_complexity, kcomplexity, previous_used_nodes=None,
        default_node_complexity:float=1.) -> Tuple[float]:
    """ Compute the general complexity of an Option with used nodes.

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
        default_node_complexity: Default individual complexity (if not given by Node).
            Default is 1.

    Returns:
        Tuple composed of the general complexity and the total saved complexity.

    """

    previous_used_nodes = previous_used_nodes if previous_used_nodes else {}

    total_complexity = 0
    total_saved_complexity = 0

    if option in used_nodes_all[option]:
        used_nodes_all[option].pop(option)

    for node in used_nodes_all[option]:
        n_used = used_nodes_all[option][node]
        n_previous_used = previous_used_nodes[node] if node in previous_used_nodes else 0

        if isinstance(node, Option):
            saved = saved_complexity(node, n_used, n_previous_used)
            if saved > 0:
                option_complexity, _ = general_complexity(node, used_nodes_all,
                    saved_complexity=saved_complexity, kcomplexity=kcomplexity,
                    previous_used_nodes=deepcopy(previous_used_nodes))
                total_saved_complexity += option_complexity * saved
        else:
            try:
                node_complexity = node.complexity
            except AttributeError:
                node_complexity = default_node_complexity
            total_complexity += node_complexity * kcomplexity(node, n_used)

        previous_used_nodes = update_sum_dict(previous_used_nodes, {node: n_used})

    return total_complexity - total_saved_complexity, total_saved_complexity
