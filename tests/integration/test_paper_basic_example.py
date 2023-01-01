# HEBGraph for explainable hierarchical reinforcement learning
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

from hebg import Action, Behavior, FeatureCondition, HEBGraph
from hebg.metrics.histograms import nodes_histograms
from hebg.metrics.complexity.complexities import learning_complexity
from hebg.requirements_graph import build_requirement_graph
from hebg.heb_graph import BEHAVIOR_SEPARATOR

from tests.examples.behaviors.report_example import Behavior0, Behavior1, Behavior2


class TestPaperBasicExamples:
    """Basic examples from the initial paper"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize variables."""

        self.actions: List[Action] = [Action(i, complexity=1) for i in range(3)]
        self.feature_conditions: List[FeatureCondition] = [
            FeatureCondition(f"feature {i}", complexity=1) for i in range(6)
        ]
        self.behaviors: List[Behavior] = [Behavior0(), Behavior1(), Behavior2()]

        self.expected_used_nodes_all: Dict[Behavior, Dict[Action, int]] = {
            self.behaviors[0]: {
                self.actions[0]: 1,
                self.actions[1]: 1,
                self.feature_conditions[0]: 1,
            },
            self.behaviors[1]: {
                self.actions[0]: 1,
                self.actions[2]: 1,
                self.behaviors[0]: 1,
                self.feature_conditions[1]: 1,
                self.feature_conditions[2]: 1,
            },
            self.behaviors[2]: {
                self.actions[0]: 1,
                self.behaviors[0]: 1,
                self.behaviors[1]: 2,
                self.feature_conditions[3]: 1,
                self.feature_conditions[4]: 1,
                self.feature_conditions[5]: 1,
            },
        }

    def test_nodes_histograms(self):
        """should give expected nodes_histograms results."""
        used_nodes_all = nodes_histograms(self.behaviors)
        check.equal(used_nodes_all, self.expected_used_nodes_all)

    def test_learning_complexity(self):
        """should give expected learning_complexity."""
        expected_learning_complexities = {
            self.behaviors[0]: 3,
            self.behaviors[1]: 6,
            self.behaviors[2]: 9,
        }
        expected_saved_complexities = {
            self.behaviors[0]: 0,
            self.behaviors[1]: 1,
            self.behaviors[2]: 12,
        }

        for behavior in self.behaviors:
            c_learning, saved_complexity = learning_complexity(
                behavior, used_nodes_all=self.expected_used_nodes_all
            )

            print(
                f"{behavior}: {c_learning}|{expected_learning_complexities[behavior]}"
                f" {saved_complexity}|{expected_saved_complexities[behavior]}"
            )
            diff_complexity = abs(c_learning - expected_learning_complexities[behavior])
            diff_saved = abs(saved_complexity - expected_saved_complexities[behavior])
            check.less(diff_complexity, 1e-14)
            check.less(diff_saved, 1e-14)

    def test_requirement_graph_edges(self):
        """should give expected requirement_graph edges."""
        expected_requirement_graph = DiGraph()
        for behavior in self.behaviors:
            expected_requirement_graph.add_node(behavior)
        expected_requirement_graph.add_edge(self.behaviors[0], self.behaviors[1])
        expected_requirement_graph.add_edge(self.behaviors[0], self.behaviors[2])
        expected_requirement_graph.add_edge(self.behaviors[1], self.behaviors[2])

        requirements_graph = build_requirement_graph(self.behaviors)
        for behavior, other_behavior in permutations(self.behaviors, 2):
            print(behavior, other_behavior)
            req_has_edge = requirements_graph.has_edge(behavior, other_behavior)
            expected_req_has_edge = expected_requirement_graph.has_edge(
                behavior, other_behavior
            )
            check.equal(req_has_edge, expected_req_has_edge)

    def test_requirement_graph_levels(self):
        """should give expected requirement_graph node levels (requirement depth)."""
        expected_levels = {
            self.behaviors[0]: 0,
            self.behaviors[1]: 1,
            self.behaviors[2]: 2,
        }
        requirements_graph = build_requirement_graph(self.behaviors)
        for behavior, level in requirements_graph.nodes(data="level"):
            check.equal(level, expected_levels[behavior])

    def test_unrolled_behaviors_graphs(self):
        """should give expected unrolled_behaviors_graphs for each example behaviors."""

        def lname(*args):
            return BEHAVIOR_SEPARATOR.join([str(arg) for arg in args])

        expected_graph_0 = deepcopy(self.behaviors[0].graph)

        expected_graph_1 = HEBGraph(self.behaviors[1])
        feature_0 = FeatureCondition(lname(self.behaviors[0], "feature 0"))
        expected_graph_1.add_edge(
            feature_0, Action(0, lname(self.behaviors[0], "action 0")), index=False
        )
        expected_graph_1.add_edge(
            feature_0, Action(1, lname(self.behaviors[0], "action 1")), index=True
        )
        feature_1 = FeatureCondition("feature 1")
        feature_2 = FeatureCondition("feature 2")
        expected_graph_1.add_edge(feature_1, feature_0, index=False)
        expected_graph_1.add_edge(feature_1, feature_2, index=True)
        expected_graph_1.add_edge(feature_2, Action(0), index=False)
        expected_graph_1.add_edge(feature_2, Action(2), index=True)

        expected_graph_2 = HEBGraph(self.behaviors[2])
        feature_3 = FeatureCondition("feature 3")
        feature_4 = FeatureCondition("feature 4")
        feature_5 = FeatureCondition("feature 5")
        expected_graph_2.add_edge(feature_3, feature_4, index=False)
        expected_graph_2.add_edge(feature_3, feature_5, index=True)
        expected_graph_2.add_edge(feature_4, Action(0), index=False)

        feature_0 = FeatureCondition(
            lname(self.behaviors[1], self.behaviors[0], "feature 0")
        )
        expected_graph_2.add_edge(
            feature_0,
            Action(0, lname(self.behaviors[1], self.behaviors[0], "action 0")),
            index=False,
        )
        expected_graph_2.add_edge(
            feature_0,
            Action(1, lname(self.behaviors[1], self.behaviors[0], "action 1")),
            index=True,
        )
        feature_1 = FeatureCondition(lname(self.behaviors[1], "feature 1"))
        feature_2 = FeatureCondition(lname(self.behaviors[1], "feature 2"))
        expected_graph_2.add_edge(feature_1, feature_0, index=False)
        expected_graph_2.add_edge(feature_1, feature_2, index=True)
        expected_graph_2.add_edge(
            feature_2, Action(0, lname(self.behaviors[1], "action 0")), index=False
        )
        expected_graph_2.add_edge(
            feature_2, Action(2, lname(self.behaviors[1], "action 2")), index=True
        )

        expected_graph_2.add_edge(feature_4, feature_1, index=True)

        feature_0_0 = FeatureCondition(lname(self.behaviors[0], "feature 0"))
        expected_graph_2.add_edge(
            feature_0_0,
            Action(0, lname(self.behaviors[0], "action 0")),
            index=False,
        )
        expected_graph_2.add_edge(
            feature_0_0,
            Action(1, lname(self.behaviors[0], "action 1")),
            index=True,
        )

        expected_graph_2.add_edge(feature_5, feature_1, index=False)
        expected_graph_2.add_edge(feature_5, feature_0_0, index=True)

        expected_graph = {
            self.behaviors[0]: expected_graph_0,
            self.behaviors[1]: expected_graph_1,
            self.behaviors[2]: expected_graph_2,
        }
        for behavior in self.behaviors:
            unrolled_graph = behavior.graph.unrolled_graph
            check.is_true(is_isomorphic(unrolled_graph, expected_graph[behavior]))

            # fig, axes = plt.subplots(1, 2)
            # unrolled_graph = behavior.graph.unrolled_graph
            # unrolled_graph.draw(axes[0], draw_behaviors_hulls=True)
            # expected_graph[behavior].draw(axes[1], draw_behaviors_hulls=True)
            # plt.show()
