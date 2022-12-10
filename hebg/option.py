# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Module for base Option. """

from __future__ import annotations

from typing import TYPE_CHECKING

from hebg.graph import compute_levels
from hebg.node import Node

if TYPE_CHECKING:
    from hebg.option_graph import OptionGraph


class Option(Node):

    """Abstract class for options"""

    def __init__(self, name: str, image=None) -> None:
        super().__init__(name, "option", image=image)
        self._graph = None

    def __call__(self, observation, *args, **kwargs):
        """Use the option to get next actions.

        By default, uses the OptionGraph if it can be built.

        Args:
            observation: Observations of the environment.
            greedy: If true, the agent should act greedily.

        Returns:
            action: Action given by the option with current observation.
            option_done: True if the option is done, False otherwise.

        """
        return self.graph.__call__(observation, *args, **kwargs)

    def build_graph(self) -> OptionGraph:
        """Build the OptionGraph of this Option.

        Returns:
            The built OptionGraph.

        """
        raise NotImplementedError

    @property
    def graph(self) -> OptionGraph:
        """Access to the Option's graph.

        Only build's the graph the first time called for efficiency.

        Returns:
            This Option's OptionGraph.

        """
        if self._graph is None:
            self._graph = self.build_graph()
            compute_levels(self._graph)
        return self._graph
