import pytest_check as check

from tests.integration import (
    FundamentalBehavior,
    ThresholdFeatureCondition,
    F_A_Behavior,
)
from hebg.node import Action
from hebg.codegen import get_hebg_source


def test_a_graph_codegen():
    """(A) Fundamental behaviors (single Action node) should return action call."""
    action = Action(42)
    a_graph = FundamentalBehavior(action).graph
    source_code = get_hebg_source(a_graph)
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
    f_a_graph = behavior.graph
    source_code = get_hebg_source(f_a_graph)
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
