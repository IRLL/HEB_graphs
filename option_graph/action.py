# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

from typing import Any

import numpy as np

from option_graph.node import Node

class Action(Node):

    def __init__(self, action:Any, name:str=None, image=None) -> None:
        self.action = action
        name = name if name is not None else f"action {action}"
        super().__init__(name, image)

    def __call__(self, observation:Any) -> Any:
        return self.action
