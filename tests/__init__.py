# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2024 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""Tests for the heb_graph package."""

from typing import Protocol
from matplotlib import pyplot as plt
import networkx as nx


class Graph(Protocol):
    def draw(self, ax, pos):
        """Draw the graph on a matplotlib axes."""

    def nodes(self) -> list:
        """Return a list of nodes"""


def plot_graph(graph: Graph, **kwargs):
    _, ax = plt.subplots()
    pos = None
    if len(list(graph.nodes())) == 0:
        pos = nx.spring_layout(graph)
    graph.draw(ax, pos=pos, **kwargs)
    plt.axis("off")  # turn off axis
    plt.show()
