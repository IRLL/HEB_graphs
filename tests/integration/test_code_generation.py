from typing import Dict, Tuple, Any

import pytest
import pytest_check as check

from tests.integration import (
    FundamentalBehavior,
    ThresholdFeatureCondition,
    F_A_Behavior,
    HEBGraph,
)
from hebg.node import Action, FeatureCondition
from hebg.behavior import Behavior
from hebg.codegen import get_hebg_source


class TestABehavior:
    """(A) Fundamental behaviors (single Action node) should return action call."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.behavior = FundamentalBehavior(Action(42))

    def test_source_codegen(self):
        source_code = get_hebg_source(self.behavior.graph)
        expected_source_code = "\n".join(
            (
                "class Action42Behavior:",
                "    def __init__(self, actions:Dict[str, Action], feature_conditions: Dict[str, FeatureCondition]):",
                "        self.actions = actions",
                "        self.feature_conditions = feature_conditions",
                "    def __call__(self, observation):",
                "        return self.actions['action 42'](observation)",
            )
        )
        check.equal(source_code, expected_source_code)

    def test_exec_codegen(self):
        check_execution_for_values(self.behavior, "Action42Behavior", (1, -1))


class TestFABehavior:
    """(F-A) Feature condition should generate if/else condition."""

    @pytest.fixture(autouse=True)
    def setup(self):
        feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)
        actions = {0: Action(0), 1: Action(1)}
        self.behavior = F_A_Behavior("Is above_zero", feature_condition, actions)

    def test_source_codegen(self):
        source_code = get_hebg_source(self.behavior.graph)
        expected_source_code = "\n".join(
            (
                "class IsAboveZero:",
                "    def __init__(self, actions:Dict[str, Action], feature_conditions: Dict[str, FeatureCondition]):",
                "        self.actions = actions",
                "        self.feature_conditions = feature_conditions",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['Greater or equal to 0 ?'](observation)",
                "        if edge_index == 0:",
                "            return self.actions['action 0'](observation)",
                "        if edge_index == 1:",
                "            return self.actions['action 1'](observation)",
            )
        )

        check.equal(source_code, expected_source_code)

    def test_exec_codegen(self):
        check_execution_for_values(self.behavior, "IsAboveZero", (1, -1))


class TestFFABehavior:
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

            for action, edge_index in zip(range(2), (0, 1)):
                graph.add_edge(feature_condition_2, Action(action), index=edge_index)

            for action, edge_index in zip(range(2, 4), (0, 1)):
                graph.add_edge(feature_condition_3, Action(action), index=edge_index)

            return graph

    @pytest.fixture(autouse=True)
    def setup(self):
        self.behavior = self.F_F_A_Behavior("scalar classification ]-1,0,1[ ?")

    def test_source_codegen(self):
        source_code = get_hebg_source(self.behavior.graph)
        expected_source_code = "\n".join(
            (
                "class ScalarClassification101:",
                "    def __init__(self, actions:Dict[str, Action], feature_conditions: Dict[str, FeatureCondition]):",
                "        self.actions = actions",
                "        self.feature_conditions = feature_conditions",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['Greater or equal to 0 ?'](observation)",
                "        if edge_index == 0:",
                "            edge_index_1 = self.feature_conditions['Lesser or equal to 1 ?'](observation)",
                "            if edge_index_1 == 0:",
                "                return self.actions['action 0'](observation)",
                "            if edge_index_1 == 1:",
                "                return self.actions['action 1'](observation)",
                "        if edge_index == 1:",
                "            edge_index_1 = self.feature_conditions['Greater or equal to -1 ?'](observation)",
                "            if edge_index_1 == 0:",
                "                return self.actions['action 2'](observation)",
                "            if edge_index_1 == 1:",
                "                return self.actions['action 3'](observation)",
            )
        )

        check.equal(source_code, expected_source_code)

    def test_exec_codegen(self):
        check_execution_for_values(
            self.behavior, "ScalarClassification101", (2, 1, -1, -2)
        )


class TestFBBehavior:
    """(F-BA) Behaviors should only call the behavior like an action."""

    @pytest.fixture(autouse=True)
    def setup(self):
        feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)
        actions = {0: Action(0), 1: Action(1)}
        sub_behavior = F_A_Behavior("Is above_zero", feature_condition, actions)

        feature_condition = ThresholdFeatureCondition(relation="<=", threshold=1)
        actions = {0: Action(0), 1: sub_behavior}
        self.behavior = F_A_Behavior("Is between 0 and 1 ?", feature_condition, actions)

    def test_source_codegen(self):
        source_code = get_hebg_source(self.behavior.graph)
        expected_source_code = "\n".join(
            (
                "class ScalarClassification101:",
                "    def __init__(self, actions:Dict[str, Action], feature_conditions: Dict[str, FeatureCondition], behaviors: Dict[str, Behaviors]):",
                "        self.actions = actions",
                "        self.feature_conditions = feature_conditions",
                "        self.known_behaviors = behaviors"
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['Lesser or equal to 1 ?'](observation)",
                "        if edge_index == 0:",
                "            return self.actions['action 0'](observation)",
                "        if edge_index == 1:",
                "            return self.known_behaviors['Is above_zero'](observation)",
            )
        )

        check.equal(source_code, expected_source_code)

    def test_exec_codegen(self):
        check_execution_for_values(
            self.behavior, "ScalarClassification101", (-1, 0, 1, 2)
        )


def check_execution_for_values(behavior: Behavior, class_name: str, values: Tuple[Any]):
    exec(get_hebg_source(behavior.graph))
    CodeGenPolicy = locals()[class_name]

    actions = {
        node.name: node for node in behavior.graph.nodes if isinstance(node, Action)
    }
    feature_conditions = {
        node.name: node
        for node in behavior.graph.nodes
        if isinstance(node, FeatureCondition)
    }

    behavior_rebuilt = CodeGenPolicy(
        actions=actions, feature_conditions=feature_conditions
    )

    for val in values:
        check.equal(behavior(val), behavior_rebuilt(val))
