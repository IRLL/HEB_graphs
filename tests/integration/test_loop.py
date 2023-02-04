import pytest
import pytest_check as check

import networkx as nx
from hebg import HEBGraph
from hebg.unrolling import unroll_graph

from tests.examples.behaviors.loop import build_looping_behaviors

import matplotlib.pyplot as plt


class TestLoop:
    """Tests for the loop example"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.gather_wood, self.get_new_axe = build_looping_behaviors()

    def test_unroll_gather_wood(self):
        draw = False
        unrolled_graph = unroll_graph(self.gather_wood.graph)
        if draw:
            _plot_graph(unrolled_graph)

        expected_graph = nx.DiGraph()
        expected_graph.add_edge("Has axe", "Punch tree")
        expected_graph.add_edge("Has axe", "Cut tree with axe")
        expected_graph.add_edge("Has axe", "Has wood")

        # Expected sub-behavior
        expected_graph.add_edge("Has wood", "Gather wood")
        expected_graph.add_edge("Has wood", "Craft axe")
        check.is_true(nx.is_isomorphic(unrolled_graph, expected_graph))

    def test_unroll_get_new_axe(self):
        draw = False
        unrolled_graph = unroll_graph(self.get_new_axe.graph)
        if draw:
            _plot_graph(unrolled_graph)

        expected_graph = nx.DiGraph()
        expected_graph.add_edge("Has wood", "Has axe")
        expected_graph.add_edge("Has wood", "Craft new axe")

        # Expected sub-behavior
        expected_graph.add_edge("Has axe", "Punch tree")
        expected_graph.add_edge("Has axe", "Cut tree with axe")
        expected_graph.add_edge("Has axe", "Get new axe")
        check.is_true(nx.is_isomorphic(unrolled_graph, expected_graph))

    def test_unroll_gather_wood_cutting_alternatives(self):
        draw = False
        unrolled_graph = unroll_graph(
            self.gather_wood.graph,
            cut_looping_alternatives=True,
        )
        if draw:
            _plot_graph(unrolled_graph)

        expected_graph = nx.DiGraph()
        expected_graph.add_edge("Has axe", "Punch tree")
        expected_graph.add_edge("Has axe", "Cut tree with axe")
        expected_graph.add_edge("Has axe", "Has wood")

        # Expected sub-behavior
        expected_graph.add_edge("Has wood", "Punch tree")
        expected_graph.add_edge("Has wood", "Craft axe")
        check.is_true(nx.is_isomorphic(unrolled_graph, expected_graph))

    def test_unroll_get_new_axe_cutting_alternatives(self):
        draw = False
        unrolled_graph = unroll_graph(
            self.get_new_axe.graph,
            cut_looping_alternatives=True,
        )
        if draw:
            _plot_graph(unrolled_graph)

        expected_graph = nx.DiGraph()
        expected_graph.add_edge("Has wood", "Has axe")
        expected_graph.add_edge("Has wood", "Craft new axe")

        # Expected sub-behavior
        expected_graph.add_edge("Has axe", "Punch tree")
        expected_graph.add_edge("Has axe", "Cut tree with axe")
        check.is_true(nx.is_isomorphic(unrolled_graph, expected_graph))


def _plot_graph(graph: "HEBGraph"):
    _, ax = plt.subplots()
    graph.draw(ax)
    plt.show()
