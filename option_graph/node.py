# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Module for base Node classes. """

from typing import Any

class Node():

    """ Base Node class for any OptionGraph. """

    NODE_TYPES = ('action', 'feature_condition', 'option', 'empty')

    def __init__(self, name:str, node_type:str, image=None) -> None:
        self.name = name
        self.image = image
        if node_type not in self.NODE_TYPES:
            raise ValueError(f'node_type ({node_type})'
                             f'not in authorised node_types ({self.NODE_TYPES}).')
        self.type = node_type

    def __call__(self, observation:Any) -> Any:
        raise NotImplementedError


class Action(Node):

    """ Node representing an action in an OptionGraph. """

    def __init__(self, action:Any, name:str=None, image=None) -> None:
        self.action = action
        name = name if name is not None else f"action {action}"
        super().__init__(name, 'action', image=image)

    def __call__(self, observation:Any) -> Any:
        return self.action


class FeatureCondition(Node):

    """ Node representing a feature condition in an OptionGraph. """

    def __init__(self, name:str=None, image=None) -> None:
        super().__init__(name, 'feature_condition', image=image)

    def __call__(self, observation:Any) -> int:
        raise NotImplementedError


class EmptyNode(Node):

    """ Node representing an empty node in an OptionGraph. """

    def __init__(self, name:str) -> None:
        super().__init__(name, 'empty')

    def __call__(self, observation:Any) -> int:
        return int(True)
