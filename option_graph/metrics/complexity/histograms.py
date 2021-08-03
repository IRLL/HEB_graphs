# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" OptionGraph used nodes histograms computation. """

from typing import List, Dict, Tuple
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from option_graph import Option, Node
from option_graph.metrics.complexity.utils import update_sum_dict

def get_used_nodes(options:List[Option], used_nodes:Dict[str, int]=None,
        default_node_complexity:float=1., verbose=0):
    complexities, used_nodes = {}, {}

    iterator = tqdm(options, total=len(options), desc='Building options histograms') \
        if verbose > 0 else options

    for option in iterator:
        complexity, _used_nodes = get_used_nodes_single_option(option, used_nodes,
            default_node_complexity=default_node_complexity)
        complexities[str(option)] = complexity
        used_nodes[str(option)] = _used_nodes
    return complexities, used_nodes


def _get_node_complexity(node:Node, used_nodes: Dict[Node, int],
    options_in_search=None, default_node_complexity:float=1.):
    if node.type in ('action', 'feature_condition'):
        try:
            node_complexity = node.complexity
        except AttributeError:
            node_complexity = default_node_complexity
        return node_complexity, {str(node):1}
    if node.type == 'option':
        if options_in_search is not None and str(node) in options_in_search:
            return np.inf, {}
        return get_used_nodes_single_option(node, used_nodes,
                    default_node_complexity=default_node_complexity,
                    _options_in_search=deepcopy(options_in_search))
    if node.type == 'empty':
        return 0, {}
    raise ValueError(f"Unkowned node type {node.type}")

def get_used_nodes_single_option(option:Option, used_nodes:Dict[str, int]=None,
        default_node_complexity:float=1., _options_in_search=None) -> Tuple[float, dict]:

    if used_nodes is None:
        used_nodes = {}

    try:
        graph = option.graph
    except NotImplementedError:
        return 0, used_nodes

    if _options_in_search is None:
        _options_in_search = []
    _options_in_search.append(str(option))

    nodes_by_level = graph.graph['nodes_by_level']
    depth = graph.graph['depth']

    complexities = {}
    nodes_used_nodes = {}

    for level in range(depth+1)[::-1]:

        for node in nodes_by_level[level]:

            node_complexity = 0
            node_used_nodes = {}

            complexities_by_index = {}
            succ_by_index = {}
            for succ in graph.successors(node):
                succ_complexity = complexities[str(succ)]
                index = int(graph.edges[node, succ]['index'])
                try:
                    complexities_by_index[index].append(succ_complexity)
                    succ_by_index[index].append(succ)
                except KeyError:
                    complexities_by_index[index] = [succ_complexity]
                    succ_by_index[index] = [succ]

            for index, values in complexities_by_index.items():
                min_index = np.argmin(values)
                choosen_succ = succ_by_index[index][min_index]
                print(str(choosen_succ), nodes_used_nodes[str(choosen_succ)], node_used_nodes)
                node_complexity += values[min_index]
                node_used_nodes = update_sum_dict(node_used_nodes,
                    nodes_used_nodes[str(choosen_succ)])

            node_used_nodes = update_sum_dict(node_used_nodes, used_nodes)

            node_only_complexity, node_only_used_options = \
                _get_node_complexity(node, node_used_nodes,
                    default_node_complexity=default_node_complexity,
                    options_in_search=_options_in_search)
            node_complexity += node_only_complexity
            node_used_nodes = update_sum_dict(node_used_nodes, node_only_used_options)

            complexities[str(node)] = node_complexity
            nodes_used_nodes[str(node)] = node_used_nodes

    root = nodes_by_level[0][0]
    nodes_used_nodes[str(root)] = update_sum_dict(nodes_used_nodes[str(root)], {str(option): 1})
    return complexities[str(root)], nodes_used_nodes[str(root)]
