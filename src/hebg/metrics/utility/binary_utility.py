# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Simplest binary utility for HEBGraph. """

from typing import Dict, List

from hebg import Behavior


def binary_graphbased_utility(
    behavior: Behavior,
    solving_behaviors: List[Behavior],
    used_nodes: Dict[str, Dict[str, int]],
) -> bool:
    """Returns if the behavior is in the HEBGraph of any solving_behavior.

    Args:
        behavior: Behavior of which we want to compute the utility.
        solving_behaviors: list of behaviors that solves the task of interest.
        used_nodes: dictionary mapping behavior_id to nodes used in the behavior.

    Returns:
        True if the behavior in the HEBGraph of any solving_behavior. False otherwise.

    """

    for solving_behavior in solving_behaviors:
        if behavior == solving_behavior:
            return True
        if behavior in solving_behavior.graph.nodes():
            return True
        if (
            behavior in used_nodes[solving_behavior]
            and used_nodes[solving_behavior][behavior] > 0
        ):
            return True
    return False
