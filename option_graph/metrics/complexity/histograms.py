# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" OptionGraph used nodes histograms computation. """

from typing import List, Dict, Union
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from option_graph import OptionGraph, Option, Node
from option_graph.metrics.complexity.utils import update_sum_dict, init_individual_complexities

def get_used_nodes(options:Dict[str, Option], used_nodes:Dict[str, int]=None,
        individual_complexities:Union[dict, float]=1., verbose=0):
    complexities, used_nodes = {}, {}

    iterator = tqdm(options.items(), total=len(options), desc='Building options histograms') \
        if verbose > 0 else options.items()

    action_nodes, feature_nodes, _ = get_nodes_types_lists(list(options.values()))
    individual_complexities = init_individual_complexities(
        action_nodes, feature_nodes, individual_complexities)

    for option_key, option in iterator:
        if option_key not in complexities:
            complexity, _used_nodes = get_used_nodes_single_option(option, options,
                individual_complexities=individual_complexities)
            complexities[option_key] = complexity
            used_nodes[option_key] = _used_nodes
    return complexities, used_nodes

def get_nodes_types_lists(options:List[Option]):
    action_nodes, feature_nodes, option_nodes = [], [], []
    for option in options:
        try:
            graph = option.graph
        except NotImplementedError:
            continue

        for node in graph.nodes():
            node_type = graph.nodes[node]['type']
            if node_type == 'action' and node not in action_nodes:
                action_nodes.append(node)
            elif node_type == 'option' and node not in option_nodes:
                option_nodes.append(node)
            elif node_type == 'feature_check' and node not in feature_nodes:
                feature_nodes.append(node)
    return action_nodes, feature_nodes, option_nodes

def get_used_nodes_single_option(option:Option, options:Dict[str, Option],
        used_nodes:Dict[str, int]=None, individual_complexities:Union[dict, float]=1.,
        return_all_nodes=False, _options_in_search=None):

    try:
        graph = option.graph
    except NotImplementedError:
        return 0, used_nodes

    if _options_in_search is None:
        _options_in_search = []
    _options_in_search.append(str(option))

    if used_nodes is None:
        used_nodes = {}

    def _get_node_complexity(graph:OptionGraph, node:Node, used_nodes: Dict[Node, int]):
        node_type = graph.nodes[node]['type']

        if node_type in ('action', 'feature_check'):
            node_used_nodes = {node:1}
            node_complexity = individual_complexities[node]

        elif node_type == 'option':
            _option = options[node]
            if str(_option) in _options_in_search:
                node_used_nodes = {}
                node_complexity = np.inf
            else:
                node_complexity, node_used_nodes = \
                    get_used_nodes_single_option(_option, options, used_nodes,
                        _options_in_search=deepcopy(_options_in_search))

        elif node_type == 'empty':
            node_used_nodes = {}
            node_complexity = 0

        else:
            raise ValueError(f"Unkowned node type {node_type}")

        return node_complexity, node_used_nodes

    nodes_by_level = graph.graph['nodes_by_level']
    depth = graph.graph['depth']

    complexities = {}
    nodes_used_nodes = {}

    for level in range(depth+1)[::-1]:

        for node in nodes_by_level[level]:

            node_complexity = 0
            node_used_nodes = {}

            or_complexities = []
            or_succs = []
            for succ in graph.successors(node):
                succ_complexity = complexities[succ]
                if graph.edges[node, succ]['type'] == 'any':
                    or_complexities.append(succ_complexity)
                    or_succs.append(succ)
                else:
                    node_complexity += succ_complexity
                    node_used_nodes = update_sum_dict(node_used_nodes, nodes_used_nodes[succ])

            if len(or_succs) > 0:
                min_succ_id = np.argmin(or_complexities)
                min_succ = or_succs[min_succ_id]
                min_complex = or_complexities[min_succ_id]

                node_complexity += min_complex
                node_used_nodes = update_sum_dict(node_used_nodes, nodes_used_nodes[min_succ])

            node_used_nodes = update_sum_dict(node_used_nodes, used_nodes)
            node_only_complexity, node_only_used_options = \
                _get_node_complexity(graph, node, node_used_nodes)

            node_complexity += node_only_complexity
            node_used_nodes = update_sum_dict(node_used_nodes, node_only_used_options)

            complexities[node] = node_complexity
            nodes_used_nodes[node] = node_used_nodes

    root = nodes_by_level[0][0]
    nodes_used_nodes[root] = update_sum_dict(nodes_used_nodes[root], {option.option_id: 1})
    if return_all_nodes:
        return complexities, nodes_used_nodes
    return complexities[root], nodes_used_nodes[root]
