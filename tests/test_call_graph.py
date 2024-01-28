from typing import Union
from networkx import (
    DiGraph,
    draw_networkx_edges,
    draw_networkx_labels,
    draw_networkx_nodes,
)
from hebg.behavior import Behavior
from hebg.heb_graph import CallEdgeStatus, HEBGraph
from hebg.node import Action

from pytest_mock import MockerFixture

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

    def test_chain_behaviors(self, mocker: MockerFixture):
        """When sub-behaviors are chained they should be in the call graph."""

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
                graph.add_node(SubBehavior())
                return graph

        f_aa_behavior = RootBehavior()

        # Sanity check that the right action should be called.
        assert f_aa_behavior(observation=-1) == expected_action

        call_graph = f_aa_behavior.graph.call_graph
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

        expected_order = [
            "Get new axe",
            "Has wood ?",
            "Gather wood",
            "Action(Summon axe out of thin air)",
            "Has axe ?",
            "Action(Punch tree)",
        ]
        nodes_by_order = sorted(
            [(node, order) for (node, order) in call_graph.nodes(data="order")],
            key=lambda x: x[1],
        )
        assert [node for node, _order in nodes_by_order] == expected_order

        if draw:
            import matplotlib.pyplot as plt

            def status_to_color(status: Union[str, CallEdgeStatus]):
                status = CallEdgeStatus(status)
                if status is CallEdgeStatus.UNEXPLORED:
                    return "black"
                if status is CallEdgeStatus.CALLED:
                    return "green"
                if status is CallEdgeStatus.FAILURE:
                    return "red"
                raise NotImplementedError

            def call_graph_pos(call_graph: DiGraph):
                pos = {}
                amount_by_order = {}
                for node, node_data in call_graph.nodes(data=True):
                    order: int = node_data.get("order")
                    if order not in amount_by_order:
                        amount_by_order[order] = 0
                    else:
                        amount_by_order[order] += 1
                    pos[node] = [order, amount_by_order[order] / 2]
                return pos

            pos = call_graph_pos(call_graph)
            draw_networkx_nodes(call_graph, pos=pos)
            draw_networkx_labels(call_graph, pos=pos)
            draw_networkx_edges(
                call_graph,
                pos,
                edge_color=[
                    status_to_color(status)
                    for _, _, status in call_graph.edges(data="status")
                ],
                connectionstyle="arc3,rad=-0.15",
            )
            plt.axis("off")  # turn off axis
            plt.show()

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
