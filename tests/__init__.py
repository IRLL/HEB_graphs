# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2024 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""Tests for the heb_graph package."""

from typing import TYPE_CHECKING
from matplotlib import pyplot as plt
import networkx as nx

if TYPE_CHECKING:
    from hebg.heb_graph import HEBGraph


def plot_graph(graph: "HEBGraph"):
    _, ax = plt.subplots()
    pos = None
    if len(graph.roots) == 0:
        pos = nx.spring_layout(graph)
    graph.draw(ax, pos=pos)
    plt.show()
