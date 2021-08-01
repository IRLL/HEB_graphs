# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Intergration tests for basic options graphs. """

import pytest_check as check

from option_graph.node import Action, EmptyNode, FeatureCondition
from option_graph.option import Option
from option_graph.option_graph import OptionGraph

class FundamentalOption(Option):

    """ Fundamental option based on an Action. """

    def __init__(self, action: Action) -> None:
        self.action = action
        name = action.name +"_option"
        super().__init__(name, image=action.image)

    def build_graph(self) -> OptionGraph:
        graph = OptionGraph(self)
        graph.add_node(self.action)
        return graph


class ThresholdFeatureCondition(FeatureCondition):

    """ Threshold-based feature condition for scalar feature. """

    def __init__(self, relation: str=">=", threshold: float=0) -> None:
        name = f"{relation} {threshold} ?"
        self.relation = relation
        self.threshold = threshold
        super().__init__(name=name, image=None)

    def __call__(self, observation: float) -> int:
        if self.relation == ">=":
            return int(observation >= self.threshold)
        if self.relation == "<=":
            return int(observation <= self.threshold)
        raise ValueError(f"Unkowned relation: {self.relation}")

def test_a_graph():
    """ (A) Fundamental options (single action) should work properly. """
    action_id = 42
    a_graph = FundamentalOption(Action(action_id))
    check.equal(a_graph(None), action_id)

def test_f_a_graph():
    """ (F-A) Feature condition should orient path properly. """

    class F_A_Option(Option):

        """ Single feature condition option """

        def build_graph(self) -> OptionGraph:
            graph = OptionGraph(self)
            feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)
            for i in range(2):
                graph.add_edge(feature_condition, Action(i), index=i)
            return graph

    option = F_A_Option('F_A')
    check.equal(option(1), 1)
    check.equal(option(-1), 0)

def test_e_a_graph():
    """ (E-A) Empty nodes should skip to successor. """

    action_id = 42

    class E_A_Option(Option):

        """ Empty option """

        def build_graph(self) -> OptionGraph:
            graph = OptionGraph(self)
            graph.add_edge(EmptyNode("empty"), Action(action_id))
            return graph

    option = E_A_Option('E_A')
    check.equal(option(None), action_id)

def test_f_f_a_graph():
    """ (F-F-A) Feature condition should orient path properly in double chain. """

    class F_F_A_Option(Option):

        """ Double layer feature conditions option """

        def build_graph(self) -> OptionGraph:
            graph = OptionGraph(self)

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

    option = F_F_A_Option('F_F_A')
    check.equal(option(2), 0)
    check.equal(option(1), 1)
    check.equal(option(-1), 3)
    check.equal(option(-2), 2)

def test_e_f_a_graph():
    """ (E-F-A) Empty should orient path properly in chain with Feature condition. """

    class E_F_A_Option(Option):

        """ Double layer empty then feature conditions option """

        def build_graph(self) -> OptionGraph:
            graph = OptionGraph(self)
            empty = EmptyNode("empty")
            feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)

            graph.add_edge(empty, feature_condition)
            for i, edge_index in zip(range(2), (0, 1)):
                action = Action(i)
                graph.add_edge(feature_condition, action, index=edge_index)

            return graph

    option = E_F_A_Option('E_F_A')
    check.equal(option(-1), 0)
    check.equal(option(1), 1)


def test_f_e_a_graph():
    """ (F-E-A) Feature condition should orient path properly in chain with Empty. """

    class F_E_A_Option(Option):

        """ Double layer feature conditions then empty option """

        def build_graph(self) -> OptionGraph:
            graph = OptionGraph(self)

            feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)
            empty_0 = EmptyNode("empty_0")
            empty_1 = EmptyNode("empty_1")

            graph.add_edge(feature_condition, empty_0, index=int(True))
            graph.add_edge(feature_condition, empty_1, index=int(False))

            graph.add_edge(empty_0, Action(0))
            graph.add_edge(empty_1,  Action(1))

            return graph

    option = F_E_A_Option('F_E_A')
    check.equal(option(1), 0)
    check.equal(option(-1), 1)

def test_e_e_a_graph():
    """ (E-E-A) Empty should orient path properly in double chain. """

    class E_E_A_Option(Option):

        """ Double layer empty option """

        def build_graph(self) -> OptionGraph:
            graph = OptionGraph(self)

            empty_0 = EmptyNode("empty_0")
            empty_1 = EmptyNode("empty_1")

            graph.add_edge(empty_0, empty_1)
            graph.add_edge(empty_1, Action(0))

            return graph

    option = E_E_A_Option('E_E_A')
    check.equal(option(None), 0)

def test_aa_graph():
    """ (AA) Should choose between roots depending on 'any_mode'. """

    class AA_Option(Option):

        """ Double root fundamental option """

        def __init__(self, name: str, any_mode: str) -> None:
            super().__init__(name, image=None)
            self.any_mode = any_mode

        def build_graph(self) -> OptionGraph:
            graph = OptionGraph(self, any_mode=self.any_mode)

            graph.add_node(Action(0))
            graph.add_node(Action(1))

            return graph

    option = AA_Option('AA', any_mode='first')
    check.equal(option(None), 0)

    option = AA_Option('AA', any_mode='last')
    check.equal(option(None), 1)

def test_af_a_graph():
    """ (AF-A) Should choose between roots depending on 'any_mode'. """

    class AF_A_Option(Option):

        """ Double root with feature condition option """

        def __init__(self, name: str, any_mode: str) -> None:
            super().__init__(name, image=None)
            self.any_mode = any_mode

        def build_graph(self) -> OptionGraph:
            graph = OptionGraph(self, any_mode=self.any_mode)

            graph.add_node(Action(0))
            feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)

            graph.add_edge(feature_condition, Action(1), index=int(True))
            graph.add_edge(feature_condition, Action(2), index=int(False))

            return graph

    option = AF_A_Option('AF_A', any_mode='first')
    check.equal(option(1), 0)
    check.equal(option(-1), 0)

    option = AF_A_Option('AF_A', any_mode='last')
    check.equal(option(1), 1)
    check.equal(option(-1), 2)

def test_f_af_a_graph():
    """ (F-AA) Should choose between condition edges depending on 'any_mode'. """

    class AF_A_Option(Option):

        """ Double root with feature condition option """

        def __init__(self, name: str, any_mode: str) -> None:
            super().__init__(name, image=None)
            self.any_mode = any_mode

        def build_graph(self) -> OptionGraph:
            graph = OptionGraph(self, any_mode=self.any_mode)
            feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)

            graph.add_edge(feature_condition, Action(0), index=int(True))
            graph.add_edge(feature_condition, Action(1), index=int(False))
            graph.add_edge(feature_condition, Action(2), index=int(False))

            return graph

    option = AF_A_Option('AF_A', any_mode='first')
    check.equal(option(1), 0)
    check.equal(option(-1), 1)

    option = AF_A_Option('AF_A', any_mode='last')
    check.equal(option(1), 0)
    check.equal(option(-1), 2)
