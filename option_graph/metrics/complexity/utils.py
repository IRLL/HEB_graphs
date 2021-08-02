# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Utility functions for complexity computation. """

from typing import Union
from copy import deepcopy

def update_sum_dict(dict1, dict2):
    """ Give the sum of two dictionaries. """
    dict1, dict2 = deepcopy(dict1), deepcopy(dict2)
    for key, val in dict2.items():
        try:
            dict1[key] += val
        except KeyError:
            dict1[key] = val
    return dict1

def init_individual_complexities(action_nodes, feature_nodes,
    individual_complexities:Union[dict, float]=1.):
    """ Initialize a dictionary of individual complexities. """

    if isinstance(individual_complexities, (float, int)):
        individual_complexities = {
            node:individual_complexities for node in action_nodes + feature_nodes
        }

    elif isinstance(individual_complexities, dict):
        assert all(node in individual_complexities for node in action_nodes + feature_nodes), \
            "Individual complexities must be given for fundamental actions and features conditions"

    return individual_complexities
