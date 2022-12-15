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
            "    def __init__(self, actions:Dict[str, Action], feature_conditions: Dict[str, FeatureCondition]):",
            "        self.actions = actions",
            "        self.feature_conditions = feature_conditions",
            "    def __call__(self, observation):",
            "        return self.actions['action 42'](observation)",
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

            for action, edge_index in zip(range(2), (0, 1)):
                graph.add_edge(feature_condition_2, Action(action), index=edge_index)

            for action, edge_index in zip(range(2, 4), (0, 1)):
                graph.add_edge(feature_condition_3, Action(action), index=edge_index)

            return graph

    behavior = F_F_A_Behavior("scalar classification ]-1,0,1[ ?")
    graph = behavior.graph
    source_code = get_hebg_source(graph)
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
