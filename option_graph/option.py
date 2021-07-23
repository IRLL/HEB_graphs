# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Module for base Option. """

from __future__ import annotations

from typing import TYPE_CHECKING

from option_graph.graph import compute_levels
from option_graph.node import Node

if TYPE_CHECKING:
    from option_graph.option_graph import OptionGraph
    from matplotlib.axes import Axes


class Option(Node):

    """ Abstract class for options """

    def __init__(self, name:str, image=None) -> None:
        super().__init__(name, image)
        self._graph = None

    def __call__(self, observation):
        """ Use the option to get next actions.

        By default, uses the OptionGraph if it can be built.

        Args:
            observation: Observations of the environment.
            greedy: If true, the agent should act greedily.

        Returns:
            action: Action given by the option with current observation.
            option_done: True if the option is done, False otherwise.

        """
        return self.graph(observation)

    def build_graph(self) -> OptionGraph:
        """ Build the OptionGraph of this Option.

        Returns:
            The built OptionGraph.

        """
        raise NotImplementedError

    def draw_graph(self, ax:Axes, **kwargs) -> Axes:
        """ Draw this Option's graph on the given Axes.

        See OptionGraph for kwargs documentation.

        Args:
            ax: The matplotlib ax to draw on.

        Returns:
            The resulting matplotlib Axes drawn on.

        """
        return self.graph.draw(ax, **kwargs)

    @property
    def graph(self) -> OptionGraph:
        """ Access to the Option's graph.

        Only build's the graph the first time called for efficiency.

        Returns:
            This Option's OptionGraph.

        """
        if self._graph is None:
            self._graph = self.build_graph()
            compute_levels(self._graph)
        return self._graph
