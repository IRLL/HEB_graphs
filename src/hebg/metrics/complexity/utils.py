# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""Utility functions for complexity computation."""

from copy import deepcopy


def update_sum_dict(dict1: dict, dict2: dict):
    """Give the sum of two dictionaries."""
    dict1, dict2 = deepcopy(dict1), deepcopy(dict2)
    for key, val in dict2.items():
        if not isinstance(val, dict):
            try:
                dict1[key] += val
            except KeyError:
                dict1[key] = val
    return dict1
