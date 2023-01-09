from typing import Optional, Tuple, Any

import pytest
import pytest_check as check

from hebg import Action, FeatureCondition, Behavior
from hebg.codegen import GeneratedBehavior

from tests.examples.behaviors import (
    FundamentalBehavior,
    F_A_Behavior,
    F_F_A_Behavior,
    build_binary_sum_behavior,
)
from tests.examples.feature_conditions import ThresholdFeatureCondition


class TestABehavior:
    """(A) Fundamental behaviors (single Action node) should return action call."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.behavior = FundamentalBehavior(Action(42))

    def test_source_codegen(self):
        source_code = self.behavior.graph.generate_source_code()
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
        source_code = self.behavior.graph.generate_source_code()
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

    @pytest.fixture(autouse=True)
    def setup(self):
        self.behavior = F_F_A_Behavior("scalar classification ]-1,0,1[ ?")

    def test_source_codegen(self):
        source_code = self.behavior.graph.generate_source_code()
        expected_source_code = "\n".join(
            (
                "class ScalarClassification101(GeneratedBehavior):",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['Greater or equal to 0 ?'](observation)",
                "        if edge_index == 0:",
                "            edge_index_1 = self.feature_conditions['Greater or equal to -1 ?'](observation)",
                "            if edge_index_1 == 0:",
                "                return self.actions['action 0'](observation)",
                "            if edge_index_1 == 1:",
                "                return self.actions['action 1'](observation)",
                "        if edge_index == 1:",
                "            edge_index_1 = self.feature_conditions['Lesser or equal to 1 ?'](observation)",
                "            if edge_index_1 == 0:",
                "                return self.actions['action 3'](observation)",
                "            if edge_index_1 == 1:",
                "                return self.actions['action 2'](observation)",
            )
        )

        check.equal(source_code, expected_source_code)

    def test_exec_codegen(self):
        check_execution_for_values(
            self.behavior, "ScalarClassification101", (2, 1, -1, -2)
        )


class TestFBBehavior:
    """(F-BA) Behaviors should be unrolled by default if they appear only once and have a graph."""

    @pytest.fixture(autouse=True)
    def setup(self):
        feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)
        actions = {0: Action(0), 1: Action(1)}
        sub_behavior = F_A_Behavior("Is above_zero", feature_condition, actions)

        feature_condition = ThresholdFeatureCondition(relation="<=", threshold=1)
        actions = {0: Action(0), 1: sub_behavior}
        self.behavior = F_A_Behavior("Is between 0 and 1 ?", feature_condition, actions)

    def test_source_codegen(self):
        source_code = self.behavior.graph.generate_source_code()
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

    def test_source_codegen_by_ref(self):
        unrolled_source_code = self.behavior.graph.generate_source_code()
        source_code = self.behavior.graph.unrolled_graph.generate_source_code()
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
        check.equal(unrolled_source_code, expected_source_code)

    def test_source_codegen_in_all_behavior(self):
        """When the behavior is found in 'all_behaviors'
        it should used the found behavior for codegen."""
        feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)
        actions = {0: Action(0), 1: Action(1)}
        sub_behavior = F_A_Behavior("Is above_zero", feature_condition, actions)
        self.behavior.graph.all_behaviors["Is above_zero"] = sub_behavior
        source_code = self.behavior.graph.generate_source_code()
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
        """When the behavior is found in 'all_behaviors'
        it should used the found behavior for graph call."""
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


class TestNestedBehaviorReuse:
    """Behaviors should be rolled when they are used mutliple times in nested subgraphs."""

    @pytest.fixture(autouse=True)
    def setup(self):
        feature_condition = FeatureCondition(name="fc1")
        actions = {0: Action(0), 1: Action(1)}
        behavior_0 = F_A_Behavior("Behavior 0", feature_condition, actions)

        feature_condition = FeatureCondition(name="fc2")
        actions = {0: Action(0), 1: behavior_0}
        behavior_1 = F_A_Behavior("Behavior 1", feature_condition, actions)

        feature_condition = FeatureCondition(name="fc3")
        actions = {0: behavior_0, 1: behavior_1}
        self.behavior = F_A_Behavior("Behavior 2", feature_condition, actions)

    def test_nested_reuse_codegen(self):
        source_code = self.behavior.graph.generate_source_code()
        expected_source_code = "\n".join(
            (
                "class Behavior0(GeneratedBehavior):",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['fc1'](observation)",
                "        if edge_index == 0:",
                "            return self.actions['action 0'](observation)",
                "        if edge_index == 1:",
                "            return self.actions['action 1'](observation)",
                "class Behavior1(GeneratedBehavior):",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['fc2'](observation)",
                "        if edge_index == 0:",
                "            return self.actions['action 0'](observation)",
                "        if edge_index == 1:",
                "            return self.known_behaviors['Behavior0'](observation)",
                "class Behavior2(GeneratedBehavior):",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['fc3'](observation)",
                "        if edge_index == 0:",
                "            return self.known_behaviors['Behavior0'](observation)",
                "        if edge_index == 1:",
                "            return self.known_behaviors['Behavior1'](observation)",
            )
        )

        check.equal(source_code, expected_source_code)


class TestFBBBehavior:
    """(F-B-B) Behaviors should only be added once as a class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.behavior = build_binary_sum_behavior()

    def test_classes_in_codegen(self):
        source_code = self.behavior.graph.generate_source_code()
        expected_classes = [
            "IsX1InBinary",
            "IsX0InBinary",
            "IsSumOfLast3Binary2",
        ]

        for expected_class in expected_classes:
            check.equal(
                source_code.count(f"class {expected_class}"),
                1,
                msg=f"Missing or duplicated class: {expected_class}",
            )

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
    generated_source_code = behavior.graph.generate_source_code()
    exec(generated_source_code)
    CodeGenPolicy = locals()[class_name]

    actions, feature_conditions, behaviors = separate_nodes_by_type(behavior)

    _behaviors = behaviors.copy()
    while len(_behaviors) > 0:
        _, sub_behavior = _behaviors.popitem()
        if sub_behavior in behavior.graph.all_behaviors:
            sub_behavior = behavior.graph.all_behaviors[sub_behavior]
        sub_actions, sub_feature_conditions, sub_behaviors = separate_nodes_by_type(
            sub_behavior
        )
        actions.update(sub_actions)
        feature_conditions.update(sub_feature_conditions)
        _behaviors.update(sub_behaviors)
        behaviors.update(sub_behaviors)

    known_behaviors = known_behaviors if known_behaviors is not None else {}
    behaviors.update(known_behaviors)

    behavior_rebuilt = CodeGenPolicy(
        actions=actions,
        feature_conditions=feature_conditions,
        behaviors=behaviors,
    )

    for val in values:
        check.equal(behavior(val), behavior_rebuilt(val))


def separate_nodes_by_type(behavior: Behavior):
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
    return actions, feature_conditions, behaviors
