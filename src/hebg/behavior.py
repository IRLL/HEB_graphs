# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""Module for base Behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hebg.graph import compute_levels
from hebg.node import Node

if TYPE_CHECKING:
    from hebg.heb_graph import HEBGraph


class Behavior(Node):
    """Abstract class for a Behavior as Node"""

    def __init__(self, name: str, image=None) -> None:
        super().__init__(name, "behavior", image=image)
        self._graph = None

    def __call__(self, observation, *args, **kwargs):
        """Use the behavior to get next actions.

        By default, uses the HEBGraph if it can be built.

        Args:
            observation: Observations of the environment.
            greedy: If true, the agent should act greedily.

        Returns:
            action: Action given by the behavior with current observation.

        """
        return self.graph.__call__(observation, *args, **kwargs)

    def build_graph(self) -> HEBGraph:
        """Build the HEBGraph of this Behavior.

        Returns:
            The built HEBGraph.

        """
        raise NotImplementedError

    @property
    def graph(self) -> HEBGraph:
        """Access to the Behavior's graph.

        Only build's the graph the first time called for efficiency.

        Returns:
            This Behavior's HEBGraph.

        """
        if self._graph is None:
            self._graph = self.build_graph()
            compute_levels(self._graph)
        return self._graph
