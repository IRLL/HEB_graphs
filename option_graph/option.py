# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Module for base Option. """

from __future__ import annotations

from typing import TYPE_CHECKING

from option_graph.graph import compute_levels

if TYPE_CHECKING:
    from option_graph.option_graph import OptionGraph
    from matplotlib.axes import Axes


class Option():

    """ Abstract class for options """

    def __init__(self, option_id) -> None:
        self.option_id = option_id
        self._graph = None

    def __call__(self, observations, greedy: bool=False):
        """ Use the option to get next actions.

        Args:
            observations: Observations of the environment.
            greedy: If true, the agent should act greedily.

        Returns:
            actions: Actions given by the option with current observations.
            option_done: True if the option is done, False otherwise.

        """
        raise NotImplementedError

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
