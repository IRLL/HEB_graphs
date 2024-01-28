import pytest
import pytest_check as check

import networkx as nx
from hebg.unrolling import unroll_graph
from tests import plot_graph

from tests.examples.behaviors.loop_with_alternative import build_looping_behaviors
from tests.examples.behaviors.loop_without_alternative import (
    build_looping_behaviors_without_alternatives,
)


class TestLoopAlternative:
    """Tests for the loop with alternative example"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.gather_wood, self.get_new_axe = build_looping_behaviors()

    def test_unroll_gather_wood(self):
        draw = False
        unrolled_graph = unroll_graph(self.gather_wood.graph)
        if draw:
            plot_graph(unrolled_graph)

        expected_graph = nx.DiGraph()
        expected_graph.add_edge("Has axe", "Punch tree")
        expected_graph.add_edge("Has axe", "Cut tree with axe")
        expected_graph.add_edge("Has axe", "Has wood")

        # Expected sub-behavior
        expected_graph.add_edge("Has wood", "Gather wood")
        expected_graph.add_edge("Has wood", "Craft axe")
        expected_graph.add_edge("Has wood", "Summon axe out of thin air")
        check.is_true(nx.is_isomorphic(unrolled_graph, expected_graph))

    def test_unroll_get_new_axe(self):
        draw = False
        unrolled_graph = unroll_graph(self.get_new_axe.graph)
        if draw:
            plot_graph(unrolled_graph)

        expected_graph = nx.DiGraph()
        expected_graph.add_edge("Has wood", "Has axe")
        expected_graph.add_edge("Has wood", "Craft new axe")
        expected_graph.add_edge("Has wood", "Summon axe out of thin air")

        # Expected sub-behavior
        expected_graph.add_edge("Has axe", "Punch tree")
        expected_graph.add_edge("Has axe", "Cut tree with axe")
        expected_graph.add_edge("Has axe", "Get new axe")
        check.is_true(nx.is_isomorphic(unrolled_graph, expected_graph))

    def test_unroll_gather_wood_cutting_alternatives(self):
        draw = False
        unrolled_graph = unroll_graph(
            self.gather_wood.graph, cut_looping_alternatives=True
        )
        if draw:
            plot_graph(unrolled_graph)

        expected_graph = nx.DiGraph()
        expected_graph.add_edge("Has axe", "Punch tree")
        expected_graph.add_edge("Has axe", "Has wood")
        expected_graph.add_edge("Has axe", "Use axe")

        # Expected sub-behavior
        expected_graph.add_edge("Has wood", "Summon axe of out thin air")
        expected_graph.add_edge("Has wood", "Craft axe")

        check.is_true(nx.is_isomorphic(unrolled_graph, expected_graph))

    def test_unroll_get_new_axe_cutting_alternatives(self):
        draw = False
        unrolled_graph = unroll_graph(
            self.get_new_axe.graph,
            cut_looping_alternatives=True,
        )
        if draw:
            plot_graph(unrolled_graph)

        expected_graph = nx.DiGraph()
        expected_graph.add_edge("Has wood", "Has axe")
        expected_graph.add_edge("Has wood", "Craft new axe")
        expected_graph.add_edge("Has wood", "Summon axe out of thin air")

        # Expected sub-behavior
        expected_graph.add_edge("Has axe", "Punch tree")
        expected_graph.add_edge("Has axe", "Cut tree with axe")
        check.is_true(nx.is_isomorphic(unrolled_graph, expected_graph))


class TestLoopWithoutAlternative:
    """Tests for the loop without alternative example"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        (
            self.reach_forest,
            self.reach_other_zone,
            self.reach_meadow,
        ) = build_looping_behaviors_without_alternatives()

    @pytest.mark.xfail
    def test_unroll_reach_forest(self):
        draw = False
        unrolled_graph = unroll_graph(
            self.reach_forest.graph,
            add_prefix=True,
            cut_looping_alternatives=True,
        )
        if draw:
            plot_graph(unrolled_graph)

        expected_graph = nx.DiGraph()
        check.is_true(nx.is_isomorphic(unrolled_graph, expected_graph))
