# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>
# pylint: disable=arguments-differ

""" Module for building underlying requirement graphs based on a set of options. """

from __future__ import annotations
from typing import List

from networkx import DiGraph

from option_graph.option import Option
from option_graph.graph import compute_levels

def build_requirement_graph(options:List[Option]) -> DiGraph:
    """ Builds a DiGraph of the requirements induced by a list of options.

    Args:
        options_graphs: List of Option to build the requirement graph from.

    Returns:
        The requirement graph induced by the given list of options.

    """

    try:
        options_graphs = [option.graph for option in options]
    except NotImplementedError as error:
        user_msg = "All options given must be able to build an OptionGraph"
        raise NotImplementedError(user_msg) from error

    requirements_graph = DiGraph()
    for option, graph in zip(options, options_graphs):
        if option not in requirements_graph.nodes():
            requirements_graph.add_node(option)
        for node in graph.nodes():
            if isinstance(node, Option):
                if node not in options:
                    raise ValueError(f"Option {str(node)} was found in {str(option)}"
                                     f"but not given in the input list of options")
                if node not in requirements_graph.nodes():
                    requirements_graph.add_node(node)
                index = len(list(requirements_graph.successors(node))) + 1
                requirements_graph.add_edge(node, option, index=index)

    compute_levels(requirements_graph)
    return requirements_graph
