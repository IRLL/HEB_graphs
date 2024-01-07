import pytest
import pytest_check as check

import networkx as nx
from hebg import HEBGraph
from hebg.unrolling import unroll_graph

from tests.examples.behaviors.loop_without_alternative import build_looping_behaviors

import matplotlib.pyplot as plt


class TestLoop:
    """Tests for the loop example"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        (
            self.reach_forest,
            self.reach_other_zone,
            self.reach_meadow,
        ) = build_looping_behaviors()

    @pytest.mark.xfail
    def test_unroll_reach_forest(self):
        draw = False
        unrolled_graph = unroll_graph(
            self.reach_forest.graph,
            add_prefix=True,
            cut_looping_alternatives=True,
        )
        if draw:
            _plot_graph(unrolled_graph)

        expected_graph = nx.DiGraph()
        check.is_true(nx.is_isomorphic(unrolled_graph, expected_graph))


def _plot_graph(graph: "HEBGraph"):
    _, ax = plt.subplots()
    pos = None
    if len(graph.roots) == 0:
        pos = nx.spring_layout(graph)
    graph.draw(ax, pos=pos)
    plt.show()
