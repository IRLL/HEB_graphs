# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

from typing import Any

import numpy as np

from option_graph.node import Node

class FeatureCondition(Node):
    """ Feature condition. """

    def __call__(self, observation:Any) -> int:
        raise NotImplementedError
