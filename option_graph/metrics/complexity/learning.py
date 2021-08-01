# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Learning complexity. """

from option_graph.metrics.complexity import general_complexity

def learning_complexity(*args, **kwargs):
    return general_complexity(
        utility=lambda node, k, p: max(0, min(k, p + k - 1)),
        kcomplexity=lambda node, k: k,
        *args, **kwargs
    )
