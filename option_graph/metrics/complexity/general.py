# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" General complexity. """

from typing import Dict, Union
from copy import deepcopy

from option_graph import Option
from option_graph.metrics.complexity.utils import update_sum_dict, init_individual_complexities

def general_complexity(option:Option, nodes_by_type, used_nodes_all:Dict[str, Dict[str, int]],
        previous_used_nodes=None,
        utility=lambda node, k, p: 0,
        kcomplexity=lambda node, k: k,
        individual_complexities:Union[dict, float]=1.):

    action_nodes, feature_nodes, options_nodes = nodes_by_type
    previous_used_nodes = previous_used_nodes if previous_used_nodes else {}
    if not isinstance(individual_complexities, dict):
        individual_complexities = init_individual_complexities(
            action_nodes, feature_nodes, individual_complexities)

    total_complexity = 0
    saved_complexity = 0

    if option in used_nodes_all[option]:
        used_nodes_all[option].pop(option)

    for node in used_nodes_all[option]:
        n_used = used_nodes_all[option][node]
        n_previous_used = previous_used_nodes[node] if node in previous_used_nodes else 0

        if node in options_nodes:
            util = utility(node, n_used, n_previous_used)
            if util > 0:
                option_complexity, _ = general_complexity(
                    node, nodes_by_type,used_nodes_all,
                    previous_used_nodes=deepcopy(previous_used_nodes),
                    utility=utility, kcomplexity=kcomplexity
                )
                saved_complexity += option_complexity * util
        else:
            total_complexity += individual_complexities[node] * kcomplexity(node, n_used)

        previous_used_nodes = update_sum_dict(previous_used_nodes, {node: n_used})

    return total_complexity - saved_complexity, saved_complexity
