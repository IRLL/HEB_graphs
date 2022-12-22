from typing import Optional, Tuple, Any

import pytest
import pytest_check as check

from tests.integration import (
    FundamentalBehavior,
    ThresholdFeatureCondition,
    IsDivisibleFeatureCondition,
    F_A_Behavior,
    HEBGraph,
)
from hebg.node import Action, FeatureCondition
from hebg.behavior import Behavior
from hebg.codegen import GeneratedBehavior


class TestABehavior:
    """(A) Fundamental behaviors (single Action node) should return action call."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.behavior = FundamentalBehavior(Action(42))

    def test_source_codegen(self):
        source_code = self.behavior.graph.source_code
        expected_source_code = "\n".join(
            (
                "class Action42Behavior(GeneratedBehavior):",
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
        source_code = self.behavior.graph.source_code
        expected_source_code = "\n".join(
            (
                "class IsAboveZero(GeneratedBehavior):",
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
        source_code = self.behavior.graph.source_code
        expected_source_code = "\n".join(
            (
                "class ScalarClassification101(GeneratedBehavior):",
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
        source_code = self.behavior.graph.source_code
        expected_source_code = "\n".join(
            (
                "class IsAboveZero(GeneratedBehavior):",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['Greater or equal to 0 ?'](observation)",
                "        if edge_index == 0:",
                "            return self.actions['action 0'](observation)",
                "        if edge_index == 1:",
                "            return self.actions['action 1'](observation)",
                "class IsBetween0And1(GeneratedBehavior):",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['Lesser or equal to 1 ?'](observation)",
                "        if edge_index == 0:",
                "            return self.actions['action 0'](observation)",
                "        if edge_index == 1:",
                "            return self.known_behaviors['Is above_zero'](observation)",
            )
        )

        check.equal(source_code, expected_source_code)

    def test_unrolled_source_codegen(self):
        source_code = self.behavior.graph.unrolled_graph.source_code
        expected_source_code = "\n".join(
            (
                "class IsBetween0And1(GeneratedBehavior):",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['Lesser or equal to 1 ?'](observation)",
                "        if edge_index == 0:",
                "            return self.actions['action 0'](observation)",
                "        if edge_index == 1:",
                "            edge_index_1 = self.feature_conditions['Greater or equal to 0 ?'](observation)",
                "            if edge_index_1 == 0:",
                "                return self.actions['action 0'](observation)",
                "            if edge_index_1 == 1:",
                "                return self.actions['action 1'](observation)",
            )
        )

        check.equal(source_code, expected_source_code)

    def test_exec_codegen(self):
        check_execution_for_values(self.behavior, "IsBetween0And1", (-1, 0, 1, 2))


class TestFBBehaviorNameRef:
    """(F-BA) Behaviors should work with only name reference to behavior,
    but will expect behavior to be given, even when unrolled."""

    @pytest.fixture(autouse=True)
    def setup(self):
        feature_condition = ThresholdFeatureCondition(relation="<=", threshold=1)
        actions = {0: Action(0), 1: Behavior("Is above_zero")}
        self.behavior = F_A_Behavior("Is between 0 and 1 ?", feature_condition, actions)

    def test_source_codegen(self):
        source_code = self.behavior.graph.source_code
        expected_source_code = "\n".join(
            (
                "# Require 'Is above_zero' behavior to be given.",
                "class IsBetween0And1(GeneratedBehavior):",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['Lesser or equal to 1 ?'](observation)",
                "        if edge_index == 0:",
                "            return self.actions['action 0'](observation)",
                "        if edge_index == 1:",
                "            return self.known_behaviors['Is above_zero'](observation)",
            )
        )

        check.equal(source_code, expected_source_code)

    def test_source_codegen_in_all_behavior(self):
        feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)
        actions = {0: Action(0), 1: Action(1)}
        sub_behavior = F_A_Behavior("Is above_zero", feature_condition, actions)
        self.behavior.graph.all_behaviors["Is above_zero"] = sub_behavior
        source_code = self.behavior.graph.source_code
        expected_source_code = "\n".join(
            (
                "class IsAboveZero(GeneratedBehavior):",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['Greater or equal to 0 ?'](observation)",
                "        if edge_index == 0:",
                "            return self.actions['action 0'](observation)",
                "        if edge_index == 1:",
                "            return self.actions['action 1'](observation)",
                "class IsBetween0And1(GeneratedBehavior):",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['Lesser or equal to 1 ?'](observation)",
                "        if edge_index == 0:",
                "            return self.actions['action 0'](observation)",
                "        if edge_index == 1:",
                "            return self.known_behaviors['Is above_zero'](observation)",
            )
        )

        check.equal(source_code, expected_source_code)

    def test_unrolled_source_codegen(self):
        source_code = self.behavior.graph.unrolled_graph.source_code
        expected_source_code = "\n".join(
            (
                "# Require 'Is above_zero' behavior to be given.",
                "class IsBetween0And1(GeneratedBehavior):",
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
        feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)
        actions = {0: Action(0), 1: Action(1)}
        sub_behavior = F_A_Behavior("Is above_zero", feature_condition, actions)
        self.behavior.graph.all_behaviors["Is above_zero"] = sub_behavior
        check_execution_for_values(
            self.behavior,
            "IsBetween0And1",
            (-1, 0, 1, 2),
            known_behaviors={"Is above_zero": sub_behavior},
        )


class TestFBBBehavior:
    """(F-B-B) Behaviors should only be added once as a class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        feature_condition = IsDivisibleFeatureCondition(2)
        actions = {0: Action(0), 1: Action(1)}
        binary_1 = F_A_Behavior("Is x1 in binary ?", feature_condition, actions)

        feature_condition = IsDivisibleFeatureCondition(2)
        actions = {0: Action(1), 1: Action(0)}
        binary_0 = F_A_Behavior("Is x0 in binary ?", feature_condition, actions)

        feature_condition = IsDivisibleFeatureCondition(4)
        actions = {0: Action(0), 1: binary_1}
        binary_11 = F_A_Behavior("Is x11 in binary ?", feature_condition, actions)

        feature_condition = IsDivisibleFeatureCondition(4)
        actions = {0: binary_0, 1: binary_1}
        binary_10_01 = F_A_Behavior(
            "Is x01 or x10 in binary ?", feature_condition, actions
        )

        feature_condition = IsDivisibleFeatureCondition(8)
        actions = {0: binary_11, 1: binary_10_01}
        self.behavior = F_A_Behavior(
            "Is sum (of last 3 binary) 2 ?", feature_condition, actions
        )

    def test_source_codegen(self):
        source_code = self.behavior.graph.source_code
        expected_source_code = "\n".join(
            (
                "class IsX1InBinary(GeneratedBehavior):",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['Is divisible by 2 ?'](observation)",
                "        if edge_index == 0:",
                "            return self.actions['action 0'](observation)",
                "        if edge_index == 1:",
                "            return self.actions['action 1'](observation)",
                "class IsX0InBinary(GeneratedBehavior):",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['Is divisible by 2 ?'](observation)",
                "        if edge_index == 0:",
                "            return self.actions['action 1'](observation)",
                "        if edge_index == 1:",
                "            return self.actions['action 0'](observation)",
                "class IsX01OrX10InBinary(GeneratedBehavior):",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['Is divisible by 4 ?'](observation)",
                "        if edge_index == 0:",
                "            return self.known_behaviors['Is x0 in binary ?'](observation)",
                "        if edge_index == 1:",
                "            return self.known_behaviors['Is x1 in binary ?'](observation)",
                "class IsX11InBinary(GeneratedBehavior):",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['Is divisible by 4 ?'](observation)",
                "        if edge_index == 0:",
                "            return self.actions['action 0'](observation)",
                "        if edge_index == 1:",
                "            return self.known_behaviors['Is x1 in binary ?'](observation)",
                "class IsSumOfLast3Binary2(GeneratedBehavior):",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['Is divisible by 8 ?'](observation)",
                "        if edge_index == 0:",
                "            return self.known_behaviors['Is x11 in binary ?'](observation)",
                "        if edge_index == 1:",
                "            return self.known_behaviors['Is x01 or x10 in binary ?'](observation)",
            )
        )

        check.equal(source_code, expected_source_code)

    def test_unrolled_source_codegen(self):
        source_code = self.behavior.graph.unrolled_graph.source_code
        expected_source_code = "\n".join(("", ""))

        check.equal(source_code, expected_source_code)

    def test_exec_codegen(self):
        check_execution_for_values(
            self.behavior, "IsSumOfLast3Binary2", (0, 1, 3, 5, 15)
        )


def check_execution_for_values(
    behavior: Behavior,
    class_name: str,
    values: Tuple[Any],
    known_behaviors: Optional[dict] = None,
):
    exec(behavior.graph.source_code)
    CodeGenPolicy = locals()[class_name]

    actions = {
        node.name: node for node in behavior.graph.nodes if isinstance(node, Action)
    }
    feature_conditions = {
        node.name: node
        for node in behavior.graph.nodes
        if isinstance(node, FeatureCondition)
    }
    behaviors = {
        node.name: node for node in behavior.graph.nodes if isinstance(node, Behavior)
    }
    known_behaviors = known_behaviors if known_behaviors is not None else {}
    behaviors.update(known_behaviors)

    behavior_rebuilt = CodeGenPolicy(
        actions=actions,
        feature_conditions=feature_conditions,
        behaviors=behaviors,
    )

    for val in values:
        check.equal(behavior(val), behavior_rebuilt(val))
