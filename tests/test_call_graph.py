from networkx import DiGraph

from hebg.behavior import Behavior
from hebg.call_graph import CallEdgeStatus, CallGraph, CallNode, _call_graph_pos
from hebg.heb_graph import HEBGraph
from hebg.node import Action, FeatureCondition

from pytest_mock import MockerFixture
import pytest_check as check

from tests import plot_graph

from tests.examples.behaviors import F_F_A_Behavior
from tests.examples.behaviors.loop_with_alternative import build_looping_behaviors
from tests.examples.feature_conditions import ThresholdFeatureCondition


class TestCall:
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
        assert set(call_graph.call_edge_labels()) == set(expected_graph.edges())

    def test_split_on_same_fc_index(self, mocker: MockerFixture):
        """When there are multiple indexes on the same feature condition,
        a branch should be created."""

        expected_action = Action("EXPECTED", complexity=1)

        forbidden_value = "FORBIDDEN"
        forbidden_action = Action(forbidden_value, complexity=2)
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
                graph.add_edge(
                    feature_condition, Action(0, complexity=0), index=int(True)
                )
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
        assert set(call_graph.call_edge_labels()) == set(expected_graph.edges())

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
        assert set(call_graph.call_edge_labels()) == set(expected_graph.edges())

        expected_labels = {
            CallNode(0, 0): "RootBehavior",
            CallNode(0, 1): "Greater or equal to 0 ?",
            CallNode(0, 2): "SubBehavior",
            CallNode(0, 3): "Greater or equal to 0 ?",
            CallNode(0, 4): "Action(EXPECTED)",
        }
        for node, label in call_graph.nodes(data="label"):
            check.equal(label, expected_labels[node])

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
        assert set(call_graph.call_edge_labels()) == set(expected_graph.edges())

    def test_looping_goback(self):
        """Loops with alternatives should be ignored."""
        draw = False
        _gather_wood, get_axe = build_looping_behaviors()
        assert get_axe({}) == "Punch tree"

        call_graph = get_axe.graph.call_graph

        if draw:
            plot_graph(call_graph)

        expected_labels = {
            CallNode(0, 0): "Get new axe",
            CallNode(0, 1): "Has wood ?",
            CallNode(1, 2): "Action(Summon axe out of thin air)",
            CallNode(0, 2): "Gather wood",
            CallNode(0, 3): "Has axe ?",
            CallNode(2, 4): "Get new axe",
            CallNode(0, 4): "Action(Punch tree)",
        }
        for node, label in call_graph.nodes(data="label"):
            check.equal(label, expected_labels[node])

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

        assert set(call_graph.call_edge_labels()) == set(expected_graph.edges())


class TestDraw:
    """Ensures that the graph is readable even in complex situations."""

    def test_result_on_first_branch(self):
        """Resulting action should always be on the first branch."""
        draw = False
        root_behavior = Behavior("Root", complexity=20)
        call_graph = CallGraph()
        call_graph.add_root(root_behavior, None)

        nodes = [
            (CallNode(0, 1), FeatureCondition("FC1", complexity=1)),
            (CallNode(0, 2), FeatureCondition("FC2", complexity=1)),
            (CallNode(0, 3), root_behavior),
            (CallNode(1, 1), FeatureCondition("FC3", complexity=1)),
            (CallNode(1, 2), FeatureCondition("FC4", complexity=1)),
            (CallNode(1, 3), FeatureCondition("FC5", complexity=1)),
            (CallNode(1, 4), Action("A", complexity=1)),
        ]

        for node, heb_node in nodes:
            call_graph.add_node(node, heb_node, None)

        edges = [
            (CallNode(0, 0), CallNode(0, 1), CallEdgeStatus.CALLED),
            (CallNode(0, 1), CallNode(0, 2), CallEdgeStatus.CALLED),
            (CallNode(0, 2), CallNode(0, 3), CallEdgeStatus.FAILURE),
            (CallNode(0, 0), CallNode(1, 1), CallEdgeStatus.CALLED),
            (CallNode(1, 1), CallNode(1, 2), CallEdgeStatus.CALLED),
            (CallNode(1, 2), CallNode(1, 3), CallEdgeStatus.CALLED),
            (CallNode(1, 3), CallNode(1, 4), CallEdgeStatus.CALLED),
        ]

        for start, end, status in edges:
            call_graph.add_edge(start, end, status)

        expected_poses = {
            CallNode(0, 0): [0, 0],
            CallNode(1, 1): [0, -1],
            CallNode(1, 2): [0, -2],
            CallNode(1, 3): [0, -3],
            CallNode(1, 4): [0, -4],
            CallNode(0, 1): [1, -1],
            CallNode(0, 2): [1, -2],
            CallNode(0, 3): [1, -3],
        }
        if draw:
            plot_graph(call_graph)

        assert _call_graph_pos(call_graph) == expected_poses
