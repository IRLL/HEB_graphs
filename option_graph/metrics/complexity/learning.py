# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Learning complexity. """

from option_graph import Option
from option_graph.metrics.complexity.general import general_complexity

def learning_complexity(option:Option, **kwargs):
    """ Compute the learning complexity of an Option with used nodes.

    Using the number of time each node is used in its OptionGraph we compute the learning
    complexity of an option and the total saved complexity.

    Args:
        option: The Option for which we compute the learning complexity.

    Kwargs:
        See :func:`option_graph.metrics.complexity.general.general_complexity`.

    Returns:
        Tuple composed of the learning complexity and the total saved complexity.

    """
    return general_complexity(
        option=option,
        saved_complexity=lambda node, k, p: max(0, min(k, p + k - 1)),
        kcomplexity=lambda node, k: k,
        **kwargs
    )
