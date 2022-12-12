# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Module for base Node classes. """

import dis
from typing import Any, List

import numpy as np


def bytecode_complexity(obj):
    """Compute the number of instructions in the bytecode of a given obj."""
    return len(list(dis.get_instructions(obj)))


class Node:

    """Base Node class for any HEBGraph."""

    NODE_TYPES = ("action", "feature_condition", "option", "empty")

    def __init__(
        self,
        name: str,
        node_type: str,
        complexity: int = None,
        image=None,
    ) -> None:
        self.name = name
        self.image = image
        if node_type not in self.NODE_TYPES:
            raise ValueError(
                f"node_type ({node_type})"
                f"not in authorised node_types ({self.NODE_TYPES})."
            )
        self.type = node_type
        if complexity is not None:
            self.complexity = complexity
        else:
            self.complexity = bytecode_complexity(self.__init__)
            self.complexity += bytecode_complexity(self.__call__)

    def __call__(self, observation: Any) -> Any:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name

    def __eq__(self, o: object) -> bool:
        return self.name == str(o)

    def __hash__(self) -> int:
        return self.name.__hash__()

    def __repr__(self) -> str:
        return self.name


class Action(Node):

    """Node representing an action in an HEBGraph."""

    def __init__(self, action: Any, name: str = None, **kwargs) -> None:
        self.action = action
        super().__init__(self._get_name(name), "action", **kwargs)

    def _get_name(self, name):
        """Get the default name of the action if None is given."""
        return f"action {self.action}" if name is None else name

    def __call__(self, observation: Any) -> Any:
        return self.action


class StochasticAction(Action):

    """Node representing a stochastic choice between actions in an HEBGraph."""

    def __init__(
        self, actions: List[Action], probs: list, name: str, image=None
    ) -> None:
        super().__init__(actions, name=name, image=image)
        self.probs = probs

    def __call__(self, observation):
        selected_action = np.random.choice(self.action, p=self.probs)
        return selected_action(observation)


class FeatureCondition(Node):

    """Node representing a feature condition in an HEBGraph."""

    def __init__(self, name: str = None, **kwargs) -> None:
        super().__init__(name, "feature_condition", **kwargs)

    def __call__(self, observation: Any) -> int:
        raise NotImplementedError


class EmptyNode(Node):

    """Node representing an empty node in an HEBGraph."""

    def __init__(self, name: str) -> None:
        super().__init__(name, "empty")

    def __call__(self, observation: Any) -> int:
        return int(True)
