import pytest
import pytest_check as check

import networkx as nx
from hebg.unrolling import unroll_graph
from tests import plot_graph

from tests.examples.behaviors.loop_with_alternative import build_looping_behaviors
from tests.examples.behaviors.loop_without_alternative import (
    build_looping_behaviors_without_direct_alternatives,
)


class TestLoopAlternative:
    """Tests for the loop with alternative example"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.gather_wood, self.get_new_axe = build_looping_behaviors()

    def test_unroll_gather_wood(self):
        draw = False
        unrolled_graph = unroll_graph(self.gather_wood.graph, add_prefix=True)
        if draw:
            plot_graph(unrolled_graph, draw_hulls=True, show_all_hulls=True)

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
        unrolled_graph = unroll_graph(self.get_new_axe.graph, add_prefix=True)
        if draw:
            plot_graph(unrolled_graph, draw_hulls=True, show_all_hulls=True)

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
            self.gather_wood.graph, add_prefix=True, cut_looping_alternatives=True
        )
        if draw:
            plot_graph(unrolled_graph, draw_hulls=True, show_all_hulls=True)

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
            self.get_new_axe.graph, add_prefix=True, cut_looping_alternatives=True
        )
        if draw:
            plot_graph(unrolled_graph, draw_hulls=True, show_all_hulls=True)

        expected_graph = nx.DiGraph(
            [
                ("Has wood", "Has axe"),
                ("Has wood", "Craft new axe"),
                ("Has wood", "Summon axe out of thin air"),
                # Expected sub-behavior
                ("Has axe", "Punch tree"),
                ("Has axe", "Cut tree with axe"),
            ]
        )
        check.is_true(nx.is_isomorphic(unrolled_graph, expected_graph))

    @pytest.mark.xfail
    def test_unroll_root_alternative_reach_forest(self):
        (
            reach_forest,
            _reach_other_zone,
            _reach_meadow,
        ) = build_looping_behaviors_without_direct_alternatives()
        draw = False
        unrolled_graph = unroll_graph(
            reach_forest.graph,
            add_prefix=True,
            cut_looping_alternatives=True,
        )
        if draw:
            plot_graph(unrolled_graph, draw_hulls=True, show_all_hulls=True)

        expected_graph = nx.DiGraph(
            [
                # ("Root", "Is in other zone ?"),
                # ("Root", "Is in meadow ?"),
                ("Is in other zone ?", "Reach other zone"),
                ("Is in other zone ?", "Go to forest"),
                ("Is in meadow ?", "Go to forest"),
                ("Is in meadow ?", "Reach meadow>Is in other zones ?"),
                ("Reach meadow>Is in other zone ?", "Reach meadow>Reach other zone"),
                ("Reach meadow>Is in other zone ?", "Reach meadow>Go to forest"),
            ]
        )
        check.is_true(nx.is_isomorphic(unrolled_graph, expected_graph))
