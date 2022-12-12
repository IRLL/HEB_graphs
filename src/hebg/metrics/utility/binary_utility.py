# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Simplest binary utility for HEBGraph. """

from typing import Dict, List

from hebg import Option


def binary_graphbased_utility(
    option: Option, solving_options: List[Option], used_nodes: Dict[str, Dict[str, int]]
) -> bool:
    """Returns if the option in the option graph of any solving_option.

    Args:
        option: option of which we want to compute the utility.
        solving_options: list of options that solves the task of interest.
        used_nodes: dictionary mapping option_id to nodes used in the option.

    Returns:
        True if the option in the option graph of any solving_option. False otherwise.

    """

    for solving_option in solving_options:
        if option == solving_option:
            return True
        if option in solving_option.graph.nodes():
            return True
        if (
            option in used_nodes[solving_option]
            and used_nodes[solving_option][option] > 0
        ):
            return True
    return False
