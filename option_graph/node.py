# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

from typing import Any

import numpy as np

class Node():

    def __init__(self, name:str, image=None) -> None:
        self.name = name
        self.image = image

    def __call__(self, observation:Any) -> Any:
        raise NotImplementedError


class EmptyNode(Node):

    def __init__(self, name:str) -> None:
        super().__init__(name, None)

    def __call__(self, observation:Any) -> None:
        return None
