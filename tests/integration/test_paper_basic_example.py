# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Integration tests for the initial paper examples. """

from typing import Dict, List
from copy import deepcopy

# import matplotlib.pyplot as plt


import pytest
import pytest_check as check

from itertools import permutations
from networkx.classes.digraph import DiGraph
from networkx import is_isomorphic

from option_graph import Action, Option, FeatureCondition, OptionGraph
from option_graph.metrics.complexity.histograms import nodes_histograms
from option_graph.metrics.complexity.complexities import learning_complexity
from option_graph.requirements_graph import build_requirement_graph
from option_graph.option_graph import OPTIONS_SEPARATOR


class TestPaperBasicExamples:
    """Basic examples from the initial paper"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize variables."""

        class Option0(Option):
            """Option 0"""

            def __init__(self) -> None:
                super().__init__("option 0")

            def build_graph(self) -> OptionGraph:
                graph = OptionGraph(self)
                feature = FeatureCondition("feature 0")
                graph.add_edge(feature, Action(0), index=False)
                graph.add_edge(feature, Action(1), index=True)
                return graph

        class Option1(Option):
            """Option 1"""

            def __init__(self) -> None:
                super().__init__("option 1")

            def build_graph(self) -> OptionGraph:
                graph = OptionGraph(self)
                feature_1 = FeatureCondition("feature 1")
                feature_2 = FeatureCondition("feature 2")
                graph.add_edge(feature_1, Option0(), index=False)
                graph.add_edge(feature_1, feature_2, index=True)
                graph.add_edge(feature_2, Action(0), index=False)
                graph.add_edge(feature_2, Action(2), index=True)
                return graph

        class Option2(Option):
            """Option 2"""

            def __init__(self) -> None:
                super().__init__("option 2")

            def build_graph(self) -> OptionGraph:
                graph = OptionGraph(self)
                feature_3 = FeatureCondition("feature 3")
                feature_4 = FeatureCondition("feature 4")
                feature_5 = FeatureCondition("feature 5")
                graph.add_edge(feature_3, feature_4, index=False)
                graph.add_edge(feature_3, feature_5, index=True)
                graph.add_edge(feature_4, Action(0), index=False)
                graph.add_edge(feature_4, Option1(), index=True)
                graph.add_edge(feature_5, Option1(), index=False)
                graph.add_edge(feature_5, Option0(), index=True)
                return graph

        self.actions: List[Action] = [Action(i) for i in range(3)]
        self.feature_conditions: List[FeatureCondition] = [
            FeatureCondition(f"feature {i}") for i in range(6)
        ]
        self.options: List[Option] = [Option0(), Option1(), Option2()]
        self.expected_used_nodes_all: Dict[Option, Dict[Action, int]] = {
            self.options[0]: {
                self.actions[0]: 1,
                self.actions[1]: 1,
                self.feature_conditions[0]: 1,
            },
            self.options[1]: {
                self.actions[0]: 1,
                self.actions[2]: 1,
                self.options[0]: 1,
                self.feature_conditions[1]: 1,
                self.feature_conditions[2]: 1,
            },
            self.options[2]: {
                self.actions[0]: 1,
                self.options[0]: 1,
                self.options[1]: 2,
                self.feature_conditions[3]: 1,
                self.feature_conditions[4]: 1,
                self.feature_conditions[5]: 1,
            },
        }

    def test_nodes_histograms(self):
        """should give expected nodes_histograms results."""
        used_nodes_all = nodes_histograms(self.options)
        check.equal(used_nodes_all, self.expected_used_nodes_all)

    def test_learning_complexity(self):
        """should give expected learning_complexity."""
        expected_learning_complexities = {
            self.options[0]: 3.0555555555555554,
            self.options[1]: 6.944444444444444,
            self.options[2]: 10.55555555555555,
        }
        expected_saved_complexities = {
            self.options[0]: 0.0,
            self.options[1]: 0.0,
            self.options[2]: 10.0,
        }

        for option in self.options:
            c_learning, saved_complexity = learning_complexity(
                option, used_nodes_all=self.expected_used_nodes_all
            )

            print(
                f"{option}: {c_learning}|{expected_learning_complexities[option]}"
                f" {saved_complexity}|{expected_saved_complexities[option]}"
            )
            diff_complexity = abs(c_learning - expected_learning_complexities[option])
            diff_saved = abs(saved_complexity - expected_saved_complexities[option])
            check.less(diff_complexity, 1e-14)
            check.less(diff_saved, 1e-14)

    def test_requirement_graph_edges(self):
        """should give expected requirement_graph edges."""
        expected_requirement_graph = DiGraph()
        for option in self.options:
            expected_requirement_graph.add_node(option)
        expected_requirement_graph.add_edge(self.options[0], self.options[1])
        expected_requirement_graph.add_edge(self.options[0], self.options[2])
        expected_requirement_graph.add_edge(self.options[1], self.options[2])

        requirements_graph = build_requirement_graph(self.options)
        for option, other_option in permutations(self.options, 2):
            print(option, other_option)
            req_has_edge = requirements_graph.has_edge(option, other_option)
            expected_req_has_edge = expected_requirement_graph.has_edge(
                option, other_option
            )
            check.equal(req_has_edge, expected_req_has_edge)

    def test_requirement_graph_levels(self):
        """should give expected requirement_graph node levels (requirement depth)."""
        expected_levels = {self.options[0]: 0, self.options[1]: 1, self.options[2]: 2}
        requirements_graph = build_requirement_graph(self.options)
        for option, level in requirements_graph.nodes(data="level"):
            check.equal(level, expected_levels[option])

    def test_unrolled_options_graphs(self):
        """should give expected unrolled_options_graphs for each example options."""

        def lname(*args):
            return OPTIONS_SEPARATOR.join([str(arg) for arg in args])

        expected_graph_0 = deepcopy(self.options[0].graph)

        expected_graph_1 = OptionGraph(self.options[1])
        feature_0 = FeatureCondition(lname(self.options[0], "feature 0"))
        expected_graph_1.add_edge(
            feature_0, Action(0, lname(self.options[0], "action 0")), index=False
        )
        expected_graph_1.add_edge(
            feature_0, Action(1, lname(self.options[0], "action 1")), index=True
        )
        feature_1 = FeatureCondition("feature 1")
        feature_2 = FeatureCondition("feature 2")
        expected_graph_1.add_edge(feature_1, feature_0, index=False)
        expected_graph_1.add_edge(feature_1, feature_2, index=True)
        expected_graph_1.add_edge(feature_2, Action(0), index=False)
        expected_graph_1.add_edge(feature_2, Action(2), index=True)

        expected_graph_2 = OptionGraph(self.options[2])
        feature_3 = FeatureCondition("feature 3")
        feature_4 = FeatureCondition("feature 4")
        feature_5 = FeatureCondition("feature 5")
        expected_graph_2.add_edge(feature_3, feature_4, index=False)
        expected_graph_2.add_edge(feature_3, feature_5, index=True)
        expected_graph_2.add_edge(feature_4, Action(0), index=False)

        feature_0 = FeatureCondition(
            lname(self.options[1], self.options[0], "feature 0")
        )
        expected_graph_2.add_edge(
            feature_0,
            Action(0, lname(self.options[1], self.options[0], "action 0")),
            index=False,
        )
        expected_graph_2.add_edge(
            feature_0,
            Action(1, lname(self.options[1], self.options[0], "action 1")),
            index=True,
        )
        feature_1 = FeatureCondition(lname(self.options[1], "feature 1"))
        feature_2 = FeatureCondition(lname(self.options[1], "feature 2"))
        expected_graph_2.add_edge(feature_1, feature_0, index=False)
        expected_graph_2.add_edge(feature_1, feature_2, index=True)
        expected_graph_2.add_edge(
            feature_2, Action(0, lname(self.options[1], "action 0")), index=False
        )
        expected_graph_2.add_edge(
            feature_2, Action(2, lname(self.options[1], "action 2")), index=True
        )

        expected_graph_2.add_edge(feature_4, feature_1, index=True)

        feature_0_0 = FeatureCondition(lname(self.options[0], "feature 0"))
        expected_graph_2.add_edge(
            feature_0_0,
            Action(0, lname(self.options[0], "action 0")),
            index=False,
        )
        expected_graph_2.add_edge(
            feature_0_0,
            Action(1, lname(self.options[0], "action 1")),
            index=True,
        )

        expected_graph_2.add_edge(feature_5, feature_1, index=False)
        expected_graph_2.add_edge(feature_5, feature_0_0, index=True)

        expected_graph = {
            self.options[0]: expected_graph_0,
            self.options[1]: expected_graph_1,
            self.options[2]: expected_graph_2,
        }
        for option in self.options:

            check.is_true(
                is_isomorphic(option.graph.unrolled_graph, expected_graph[option])
            )

            # fig, axes = plt.subplots(1, 2)
            # unrolled_graph = option.graph.unrolled_graph
            # unrolled_graph.draw(axes[0], draw_options_hulls=True)
            # expected_graph[option].draw(axes[1], draw_options_hulls=True)
            # plt.show()
