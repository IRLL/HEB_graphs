# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Intergration tests for basic HEBGraphs. """

import pytest_check as check

from hebg.node import Action, EmptyNode, FeatureCondition
from hebg.behavior import Behavior
from hebg.heb_graph import HEBGraph

from tests.integration import (
    FundamentalBehavior,
    ThresholdFeatureCondition,
    F_A_Behavior,
    E_A_Behavior,
)


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


def test_e_a_graph():
    """(E-A) Empty nodes should skip to successor."""
    action_id = 42
    behavior = E_A_Behavior("E_A", Action(action_id))
    check.equal(behavior(None), action_id)


def test_f_f_a_graph():
    """(F-F-A) Feature condition should orient path properly in double chain."""

    class F_F_A_Behavior(Behavior):

        """Double layer feature conditions behavior"""

        def build_graph(self) -> HEBGraph:
            graph = HEBGraph(self)

            feature_condition_1 = ThresholdFeatureCondition(relation=">=", threshold=0)
            feature_condition_2 = ThresholdFeatureCondition(relation="<=", threshold=1)
            feature_condition_3 = ThresholdFeatureCondition(relation=">=", threshold=-1)

            graph.add_edge(feature_condition_1, feature_condition_2, index=True)
            graph.add_edge(feature_condition_1, feature_condition_3, index=False)

            for action, edge_index in zip(range(2), (0, 1)):
                graph.add_edge(feature_condition_2, Action(action), index=edge_index)

            for action, edge_index in zip(range(2, 4), (0, 1)):
                graph.add_edge(feature_condition_3, Action(action), index=edge_index)

            return graph

    behavior = F_F_A_Behavior("F_F_A")
    check.equal(behavior(2), 0)
    check.equal(behavior(1), 1)
    check.equal(behavior(-1), 3)
    check.equal(behavior(-2), 2)


def test_e_f_a_graph():
    """(E-F-A) Empty should orient path properly in chain with Feature condition."""

    class E_F_A_Behavior(Behavior):

        """Double layer empty then feature conditions behavior"""

        def build_graph(self) -> HEBGraph:
            graph = HEBGraph(self)
            empty = EmptyNode("empty")
            feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)

            graph.add_edge(empty, feature_condition)
            for i, edge_index in zip(range(2), (0, 1)):
                action = Action(i)
                graph.add_edge(feature_condition, action, index=edge_index)

            return graph

    behavior = E_F_A_Behavior("E_F_A")
    check.equal(behavior(-1), 0)
    check.equal(behavior(1), 1)


def test_f_e_a_graph():
    """(F-E-A) Feature condition should orient path properly in chain with Empty."""

    class F_E_A_Behavior(Behavior):

        """Double layer feature conditions then empty behavior"""

        def build_graph(self) -> HEBGraph:
            graph = HEBGraph(self)

            feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)
            empty_0 = EmptyNode("empty_0")
            empty_1 = EmptyNode("empty_1")

            graph.add_edge(feature_condition, empty_0, index=int(True))
            graph.add_edge(feature_condition, empty_1, index=int(False))

            graph.add_edge(empty_0, Action(0))
            graph.add_edge(empty_1, Action(1))

            return graph

    behavior = F_E_A_Behavior("F_E_A")
    check.equal(behavior(1), 0)
    check.equal(behavior(-1), 1)


def test_e_e_a_graph():
    """(E-E-A) Empty should orient path properly in double chain."""

    class E_E_A_Behavior(Behavior):

        """Double layer empty behavior"""

        def build_graph(self) -> HEBGraph:
            graph = HEBGraph(self)

            empty_0 = EmptyNode("empty_0")
            empty_1 = EmptyNode("empty_1")

            graph.add_edge(empty_0, empty_1)
            graph.add_edge(empty_1, Action(0))

            return graph

    behavior = E_E_A_Behavior("E_E_A")
    check.equal(behavior(None), 0)


def test_aa_graph():
    """(AA) Should choose between roots depending on 'any_mode'."""

    class AA_Behavior(Behavior):

        """Double root fundamental behavior"""

        def __init__(self, name: str, any_mode: str) -> None:
            super().__init__(name, image=None)
            self.any_mode = any_mode

        def build_graph(self) -> HEBGraph:
            graph = HEBGraph(self, any_mode=self.any_mode)

            graph.add_node(Action(0))
            graph.add_node(Action(1))

            return graph

    behavior = AA_Behavior("AA", any_mode="first")
    check.equal(behavior(None), 0)

    behavior = AA_Behavior("AA", any_mode="last")
    check.equal(behavior(None), 1)


def test_af_a_graph():
    """(AF-A) Should choose between roots depending on 'any_mode'."""

    class AF_A_Behavior(Behavior):

        """Double root with feature condition behavior"""

        def __init__(self, name: str, any_mode: str) -> None:
            super().__init__(name, image=None)
            self.any_mode = any_mode

        def build_graph(self) -> HEBGraph:
            graph = HEBGraph(self, any_mode=self.any_mode)

            graph.add_node(Action(0))
            feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)

            graph.add_edge(feature_condition, Action(1), index=int(True))
            graph.add_edge(feature_condition, Action(2), index=int(False))

            return graph

    behavior = AF_A_Behavior("AF_A", any_mode="first")
    check.equal(behavior(1), 0)
    check.equal(behavior(-1), 0)

    behavior = AF_A_Behavior("AF_A", any_mode="last")
    check.equal(behavior(1), 1)
    check.equal(behavior(-1), 2)


def test_f_af_a_graph():
    """(F-AA) Should choose between condition edges depending on 'any_mode'."""

    class AF_A_Behavior(Behavior):

        """Double root with feature condition behavior"""

        def __init__(self, name: str, any_mode: str) -> None:
            super().__init__(name, image=None)
            self.any_mode = any_mode

        def build_graph(self) -> HEBGraph:
            graph = HEBGraph(self, any_mode=self.any_mode)
            feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)

            graph.add_edge(feature_condition, Action(0), index=int(True))
            graph.add_edge(feature_condition, Action(1), index=int(False))
            graph.add_edge(feature_condition, Action(2), index=int(False))

            return graph

    behavior = AF_A_Behavior("AF_A", any_mode="first")
    check.equal(behavior(1), 0)
    check.equal(behavior(-1), 1)

    behavior = AF_A_Behavior("AF_A", any_mode="last")
    check.equal(behavior(1), 0)
    check.equal(behavior(-1), 2)
