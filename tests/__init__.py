# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2024 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""Tests for the heb_graph package."""

from typing import Protocol
from matplotlib import pyplot as plt


class Graph(Protocol):
    def draw(self, ax, pos):
        """Draw the graph on a matplotlib axes."""

    def nodes(self) -> list:
        """Return a list of nodes"""


def plot_graph(graph: Graph, **kwargs):
    _, ax = plt.subplots()
    graph.draw(ax, **kwargs)
    plt.axis("off")  # turn off axis
    plt.show()
