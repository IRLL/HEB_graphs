# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2024 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""Behavior of HEBGraphs when called."""

import pytest
import pytest_check as check
from pytest_mock import MockerFixture

from hebg.behavior import Behavior
from hebg.heb_graph import HEBGraph
from hebg.node import Action

from tests.examples.behaviors import FundamentalBehavior, F_A_Behavior, F_F_A_Behavior
from tests.examples.behaviors.loop_with_alternative import build_looping_behaviors
from tests.examples.feature_conditions import ThresholdFeatureCondition


class TestBehavior:

    """Behavior"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize variables."""
        self.node = Behavior("behavior_name")

    def test_node_type(self):
        """should have 'behavior' as node_type."""
        check.equal(self.node.type, "behavior")

    def test_node_call(self, mocker: MockerFixture):
        """should use graph on call."""
        mocker.patch("hebg.behavior.Behavior.graph")
        self.node(None)
        check.is_true(self.node.graph.called)

    def test_build_graph(self):
        """should raise NotImplementedError when build_graph is called."""
        with pytest.raises(NotImplementedError):
            self.node.build_graph()

    def test_graph(self, mocker: MockerFixture):
        """should build graph and compute its levels if, and only if,
        the graph is not yet built.
        """
        mocker.patch("hebg.behavior.Behavior.build_graph")
        mocker.patch("hebg.behavior.compute_levels")
        self.node.graph
        check.is_true(self.node.build_graph.called)
        check.is_true(self.node.build_graph.called)

        mocker.patch("hebg.behavior.Behavior.build_graph")
        mocker.patch("hebg.behavior.compute_levels")
        self.node.graph
        check.is_false(self.node.build_graph.called)
        check.is_false(self.node.build_graph.called)


class TestPathfinding:
    def test_fundamental_behavior(self):
        """Fundamental behavior (single action) should return its action."""
        action_id = 42
        behavior = FundamentalBehavior(Action(action_id))
        check.equal(behavior(None), action_id)

    def test_feature_condition_single(self):
        """Feature condition should orient path properly."""
        feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)
        actions = {0: Action(0), 1: Action(1)}
        behavior = F_A_Behavior("F_A", feature_condition, actions)
        check.equal(behavior(1), 1)
        check.equal(behavior(-1), 0)

    def test_feature_conditions_chained(self):
        """Feature condition should orient path properly in double chain."""
        behavior = F_F_A_Behavior("F_F_A")
        check.equal(behavior(-2), 0)
        check.equal(behavior(-1), 1)
        check.equal(behavior(1), 2)
        check.equal(behavior(2), 3)

    def test_looping_resolve(self):
        """Loops with alternatives should be ignored."""
        _gather_wood, get_axe = build_looping_behaviors()
        check.equal(get_axe({}), "Punch tree")


class TestCostBehavior:
    def test_choose_root_of_lesser_cost(self):
        """Should choose root of lesser cost."""

        expected_action = "EXPECTED"

        class AAA_Behavior(Behavior):
            def __init__(self) -> None:
                super().__init__("AAA")

            def build_graph(self) -> HEBGraph:
                graph = HEBGraph(self)
                graph.add_node(Action(0, cost=2))
                graph.add_node(Action(expected_action, cost=1))
                graph.add_node(Action(2, cost=3))
                return graph

        behavior = AAA_Behavior()
        check.equal(behavior(None), expected_action)

    def test_not_path_of_least_cost(self):
        """Should choose path of larger complexity if individual costs lead to it."""

        class AF_A_Behavior(Behavior):

            """Double root with feature condition and action"""

            def __init__(self) -> None:
                super().__init__("AF_A")

            def build_graph(self) -> HEBGraph:
                graph = HEBGraph(self)

                graph.add_node(Action(0, cost=1.5))
                feature_condition = ThresholdFeatureCondition(
                    relation=">=", threshold=0, cost=1.0
                )

                graph.add_edge(feature_condition, Action(1, cost=1.0), index=int(True))
                graph.add_edge(feature_condition, Action(2, cost=1.0), index=int(False))

                return graph

        behavior = AF_A_Behavior()
        check.equal(behavior(1), 1)
        check.equal(behavior(-1), 2)
