from hebg.node import Action, EmptyNode
from hebg.behavior import Behavior
from hebg.heb_graph import HEBGraph

from tests.examples.feature_conditions import ThresholdFeatureCondition


class E_A_Behavior(Behavior):

    """Empty behavior"""

    def __init__(self, name: str, action: Action) -> None:
        super().__init__(name)
        self.action = action

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        graph.add_edge(EmptyNode("empty"), self.action)
        return graph


class E_F_A_Behavior(Behavior):

    """Double layer empty then feature conditions behavior"""

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        empty = EmptyNode("empty")
        feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)

        graph.add_edge(empty, feature_condition)
        for i, edge_index in zip(range(2), (0, 1)):
            action = Action(i)
            graph.add_edge(feature_condition, action, index=edge_index)

        return graph


class F_E_A_Behavior(Behavior):

    """Double layer feature conditions then empty behavior"""

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)

        feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)
        empty_0 = EmptyNode("empty_0")
        empty_1 = EmptyNode("empty_1")

        graph.add_edge(feature_condition, empty_0, index=int(True))
        graph.add_edge(feature_condition, empty_1, index=int(False))

        graph.add_edge(empty_0, Action(0))
        graph.add_edge(empty_1, Action(1))

        return graph


class E_E_A_Behavior(Behavior):

    """Double layer empty behavior"""

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)

        empty_0 = EmptyNode("empty_0")
        empty_1 = EmptyNode("empty_1")

        graph.add_edge(empty_0, empty_1)
        graph.add_edge(empty_1, Action(0))

        return graph
