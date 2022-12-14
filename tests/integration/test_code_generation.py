import pytest_check as check

from tests.integration import (
    FundamentalBehavior,
    ThresholdFeatureCondition,
    F_A_Behavior,
    HEBGraph,
)
from hebg.node import Action
from hebg.behavior import Behavior
from hebg.codegen import get_hebg_source


def test_a_graph_codegen():
    """(A) Fundamental behaviors (single Action node) should return action call."""
    action = Action(42)
    graph = FundamentalBehavior(action).graph
    source_code = get_hebg_source(graph)
    expected_source_code = "\n".join(
        (
            "class Action42Behavior:",
            "    def __init__(self):",
            "        self.action_42 = Action(42)",
            "    def __call__(self, observation):",
            "        return self.action_42(observation)",
        )
    )
    check.equal(source_code, expected_source_code)


def test_f_a_graph_codegen():
    """(F-A) Feature condition should generate if/else condition."""
    feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)
    actions = {0: Action(0), 1: Action(1)}
    behavior = F_A_Behavior("Is above_zero", feature_condition, actions)
    graph = behavior.graph
    source_code = get_hebg_source(graph)
    expected_source_code = "\n".join(
        (
            "class IsAboveZero:",
            "    def __init__(self):",
            '        self.greater_or_equal_to_0 = ThresholdFeatureCondition(relation=">=", threshold=0)',
            "        self.action_0 = Action(0)",
            "        self.action_1 = Action(1)",
            "    def __call__(self, observation):",
            "        if self.greater_or_equal_to_0(observation) == 0:",
            "            return self.action_0(observation)",
            "        if self.greater_or_equal_to_0(observation) == 1:",
            "            return self.action_1(observation)",
        )
    )

    check.equal(source_code, expected_source_code)


def test_f_f_a_graph_codegen():
    """(F-F-A) Chained FeatureConditions should condition should generate nested if/else."""

    class F_F_A_Behavior(Behavior):

        """Double layer feature conditions behavior"""

        def build_graph(self) -> HEBGraph:
            graph = HEBGraph(self)

            feature_condition_1 = ThresholdFeatureCondition(relation=">=", threshold=0)
            feature_condition_2 = ThresholdFeatureCondition(relation="<=", threshold=1)
            feature_condition_3 = ThresholdFeatureCondition(relation=">=", threshold=-1)

            graph.add_edge(feature_condition_1, feature_condition_2, index=False)
            graph.add_edge(feature_condition_1, feature_condition_3, index=True)

            for action, edge_index in zip(range(2), (1, 0)):
                graph.add_edge(feature_condition_2, Action(action), index=edge_index)

            for action, edge_index in zip(range(2, 4), (0, 1)):
                graph.add_edge(feature_condition_3, Action(action), index=edge_index)

            return graph

    behavior = F_F_A_Behavior("scalar classification ]-1,0,1[ ?")
    graph = behavior.graph
    source_code = get_hebg_source(graph)
    expected_source_code = "\n".join(
        (
            "class IsAboveZero:",
            "    def __init__(self):",
            '        self.greater_or_equal_to_0 = ThresholdFeatureCondition(relation=">=", threshold=0)',
            '        self.lesser_or_equal_to_1 = ThresholdFeatureCondition(relation="<=", threshold=1)',
            '        self.greater_or_equal_to_n1 = ThresholdFeatureCondition(relation=">=", threshold=-1)',
            "        self.action_0 = Action(0)",
            "        self.action_1 = Action(1)",
            "        self.action_2 = Action(2)",
            "        self.action_3 = Action(3)",
            "    def __call__(self, observation):",
            "        if self.greater_or_equal_to_0(observation) == 0:",
            "            if self.lesser_or_equal_to_1(observation) == 0:",
            "                return self.action_0(observation)",
            "            if self.lesser_or_equal_to_1(observation) == 1:",
            "                return self.action_1(observation)",
            "        if self.greater_or_equal_to_0(observation) == 1:",
            "            if self.greater_or_equal_to_n1(observation) == 0:",
            "                return self.action_2(observation)",
            "            if self.greater_or_equal_to_n1(observation) == 1:",
            "                return self.action_3(observation)",
        )
    )

    check.equal(source_code, expected_source_code)
