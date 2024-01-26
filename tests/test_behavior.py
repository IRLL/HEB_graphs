# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2024 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""Behavior of HEBGraphs when called."""

import pytest
import pytest_check as check
from pytest_mock import MockerFixture

from hebg.behavior import Behavior
from hebg.node import Action

from tests.examples.behaviors import (
    FundamentalBehavior,
    AA_Behavior,
    F_A_Behavior,
    F_F_A_Behavior,
    AF_A_Behavior,
    F_AA_Behavior,
)
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


def test_a_graph():
    """(A) Fundamental behaviors (single action) should work properly."""
    action_id = 42
    behavior = FundamentalBehavior(Action(action_id))
    check.equal(behavior(None), action_id)


def test_f_a_graph():
    """(F-A) Feature condition should orient path properly."""
    feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)
    actions = {0: Action(0), 1: Action(1)}
    behavior = F_A_Behavior("F_A", feature_condition, actions)
    check.equal(behavior(1), 1)
    check.equal(behavior(-1), 0)


def test_f_f_a_graph():
    """(F-F-A) Feature condition should orient path properly in double chain."""
    behavior = F_F_A_Behavior("F_F_A")
    check.equal(behavior(-2), 0)
    check.equal(behavior(-1), 1)
    check.equal(behavior(1), 2)
    check.equal(behavior(2), 3)


def test_aa_graph():
    """(AA) Should choose between roots depending on 'any_mode'."""
    behavior = AA_Behavior("AA", any_mode="first")
    check.equal(behavior(None), 0)

    behavior = AA_Behavior("AA", any_mode="last")
    check.equal(behavior(None), 1)


def test_af_a_graph():
    """(AF-A) Should choose between roots depending on 'any_mode'."""
    behavior = AF_A_Behavior("AF_A", any_mode="first")
    check.equal(behavior(1), 0)
    check.equal(behavior(-1), 0)

    behavior = AF_A_Behavior("AF_A", any_mode="last")
    check.equal(behavior(1), 1)
    check.equal(behavior(-1), 2)


def test_f_af_a_graph():
    """(F-AA) Should choose between condition edges depending on 'any_mode'."""
    behavior = F_AA_Behavior("F_AA", any_mode="first")
    check.equal(behavior(1), 0)
    check.equal(behavior(-1), 1)

    behavior = F_AA_Behavior("F_AA", any_mode="last")
    check.equal(behavior(1), 0)
    check.equal(behavior(-1), 2)
