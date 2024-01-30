from networkx import DiGraph

from hebg.behavior import Behavior
from hebg.heb_graph import HEBGraph
from hebg.node import Action

from pytest_mock import MockerFixture
import pytest_check as check

from tests import plot_graph

from tests.examples.behaviors import F_F_A_Behavior
from tests.examples.behaviors.loop_with_alternative import build_looping_behaviors
from tests.examples.feature_conditions import ThresholdFeatureCondition


class TestCallGraph:
    """Ensure that the call graph is faithful for debugging and efficient breadth first search."""

    def test_call_stack_without_branches(self):
        """When there is no branches, the graph should be a simple sequence of the call stack."""
        f_f_a_behavior = F_F_A_Behavior()

        draw = False
        if draw:
            plot_graph(f_f_a_behavior.graph.unrolled_graph)
        f_f_a_behavior(observation=-2)

        expected_graph = DiGraph(
            [
                ("F_F_A", "Greater or equal to 0 ?"),
                ("Greater or equal to 0 ?", "Greater or equal to -1 ?"),
                ("Greater or equal to -1 ?", "Action(0)"),
            ]
        )

        call_graph = f_f_a_behavior.graph.call_graph
        assert set(call_graph.edges()) == set(expected_graph.edges())

    def test_split_on_same_fc_index(self, mocker: MockerFixture):
        """When there are multiple indexes on the same feature condition,
        a branch should be created."""

        expected_action = Action("EXPECTED", cost=1)

        forbidden_value = "FORBIDDEN"
        forbidden_action = Action(forbidden_value, cost=2)
        forbidden_action.__call__ = mocker.MagicMock(return_value=forbidden_value)

        class F_AA_Behavior(Behavior):

            """Feature condition with mutliple actions on same index."""

            def __init__(self) -> None:
                super().__init__("F_AA")

            def build_graph(self) -> HEBGraph:
                graph = HEBGraph(self)
                feature_condition = ThresholdFeatureCondition(
                    relation=">=", threshold=0
                )
                graph.add_edge(feature_condition, Action(0, cost=0), index=int(True))
                graph.add_edge(feature_condition, forbidden_action, index=int(False))
                graph.add_edge(feature_condition, expected_action, index=int(False))

                return graph

        f_aa_behavior = F_AA_Behavior()
        draw = False
        if draw:
            plot_graph(f_aa_behavior.graph.unrolled_graph)

        # Sanity check that the right action should be called and not the forbidden one.
        assert f_aa_behavior(observation=-1) == expected_action.action
        forbidden_action.__call__.assert_not_called()

        # Graph should have the good split
        call_graph = f_aa_behavior.graph.call_graph
        expected_graph = DiGraph(
            [
                ("F_AA", "Greater or equal to 0 ?"),
                ("Greater or equal to 0 ?", "Action(EXPECTED)"),
                ("Greater or equal to 0 ?", "Action(FORBIDDEN)"),
            ]
        )
        assert set(call_graph.edges()) == set(expected_graph.edges())

    def test_multiple_call_to_same_fc(self, mocker: MockerFixture):
        """Call graph should allow for the same feature condition
        to be called multiple times in the same branch (in different behaviors)."""
        expected_action = Action("EXPECTED")
        unexpected_action = Action("UNEXPECTED")

        feature_condition_call = mocker.patch(
            "tests.examples.feature_conditions.ThresholdFeatureCondition.__call__",
            return_value=True,
        )
        feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)

        class SubBehavior(Behavior):
            def __init__(self) -> None:
                super().__init__("SubBehavior")

            def build_graph(self) -> HEBGraph:
                graph = HEBGraph(self)
                graph.add_edge(feature_condition, expected_action, index=int(True))
                graph.add_edge(feature_condition, unexpected_action, index=int(False))
                return graph

        class RootBehavior(Behavior):

            """Feature condition with mutliple actions on same index."""

            def __init__(self) -> None:
                super().__init__("RootBehavior")

            def build_graph(self) -> HEBGraph:
                graph = HEBGraph(self)
                graph.add_edge(feature_condition, SubBehavior(), index=int(True))
                graph.add_edge(feature_condition, unexpected_action, index=int(False))

                return graph

        root_behavior = RootBehavior()
        draw = False
        if draw:
            plot_graph(root_behavior.graph.unrolled_graph)

        # Sanity check that the right action should be called and not the forbidden one.
        assert root_behavior(observation=2) == expected_action.action

        # Feature condition should only be called once on the same input
        assert len(feature_condition_call.call_args_list) == 1

        # Graph should have the good split
        call_graph = root_behavior.graph.call_graph
        expected_graph = DiGraph(
            [
                ("RootBehavior", "Greater or equal to 0 ?"),
                ("Greater or equal to 0 ?", "SubBehavior"),
                ("SubBehavior", "Greater or equal to 0 ?"),
                ("Greater or equal to 0 ?", "Action(EXPECTED)"),
            ]
        )
        assert set(call_graph.edges()) == set(expected_graph.edges())

        expected_calls_order = {
            "RootBehavior": [0],
            "Greater or equal to 0 ?": [1, 3],
            "SubBehavior": [2],
            "Action(EXPECTED)": [4],
        }
        for node, node_calls_order in call_graph.nodes(data="calls_order"):
            check.equal(node_calls_order, expected_calls_order[node])

    def test_chain_behaviors(self, mocker: MockerFixture):
        """When sub-behaviors with a graph are called recursively,
        the call graph should still find their nodes."""

        expected_action = "EXPECTED"

        class DummyBehavior(Behavior):
            __call__ = mocker.MagicMock(return_value=expected_action)

        class SubBehavior(Behavior):
            def __init__(self) -> None:
                super().__init__("SubBehavior")

            def build_graph(self) -> HEBGraph:
                graph = HEBGraph(self)
                graph.add_node(DummyBehavior("Dummy"))
                return graph

        class RootBehavior(Behavior):
            def __init__(self) -> None:
                super().__init__("RootBehavior")

            def build_graph(self) -> HEBGraph:
                graph = HEBGraph(self)
                graph.add_node(Behavior("SubBehavior"))
                return graph

        sub_behavior = SubBehavior()

        root_behavior = RootBehavior()
        root_behavior.graph.all_behaviors["SubBehavior"] = sub_behavior

        # Sanity check that the right action should be called.
        assert root_behavior(observation=-1) == expected_action

        call_graph = root_behavior.graph.call_graph
        expected_graph = DiGraph(
            [
                ("RootBehavior", "SubBehavior"),
                ("SubBehavior", "Dummy"),
            ]
        )
        assert set(call_graph.edges()) == set(expected_graph.edges())

    def test_looping_goback(self):
        """Loops with alternatives should be ignored."""
        draw = False
        _gather_wood, get_axe = build_looping_behaviors()
        assert get_axe({}) == "Punch tree"

        call_graph = get_axe.graph.call_graph

        if draw:
            plot_graph(call_graph)

        expected_order = [
            "Get new axe",
            "Has wood ?",
            "Gather wood",
            "Action(Summon axe out of thin air)",
            "Has axe ?",
            "Action(Punch tree)",
        ]
        nodes_by_order = sorted(
            [
                (node, order)
                for (node, order) in call_graph.nodes(data="exploration_order")
            ],
            key=lambda x: x[1],
        )
        assert [node for node, _order in nodes_by_order] == expected_order

        expected_graph = DiGraph(
            [
                ("Get new axe", "Has wood ?"),
                ("Has wood ?", "Action(Summon axe out of thin air)"),
                ("Has wood ?", "Gather wood"),
                ("Gather wood", "Has axe ?"),
                ("Has axe ?", "Get new axe"),
                ("Has axe ?", "Action(Punch tree)"),
            ]
        )

        assert set(call_graph.edges()) == set(expected_graph.edges())
