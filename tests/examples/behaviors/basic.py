from typing import Dict

from hebg.node import Action, FeatureCondition
from hebg.behavior import Behavior
from hebg.heb_graph import HEBGraph

from tests.examples.feature_conditions import ThresholdFeatureCondition


class FundamentalBehavior(Behavior):
    """Fundamental behavior based on an Action."""

    def __init__(self, action: Action) -> None:
        self.action = action
        name = action.name + "_behavior"
        super().__init__(name, image=action.image)

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        graph.add_node(self.action)
        return graph


class AA_Behavior(Behavior):
    """Double root fundamental behavior"""

    def __init__(self, name: str, any_mode: str) -> None:
        super().__init__(name, image=None)
        self.any_mode = any_mode

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self, any_mode=self.any_mode)

        graph.add_node(Action(0))
        graph.add_node(Action(1))

        return graph


class F_A_Behavior(Behavior):
    """Single feature condition behavior"""

    def __init__(
        self,
        name: str,
        feature_condition: FeatureCondition,
        actions: Dict[int, Action],
    ) -> None:
        """Single feature condition behavior

        Args:
            name (str): Name of the behavior.
            feature_condition (FeatureCondition): Feature_condition used in behavior.
            actions (Dict[int, Action]): Mapping from feature_condition output to actions.
        """
        super().__init__(name)
        self.actions = actions
        self.feature_condition = feature_condition

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        for fc_output, action in self.actions.items():
            graph.add_edge(self.feature_condition, action, index=fc_output)
        return graph


class F_F_A_Behavior(Behavior):
    """Double layer feature conditions behavior"""

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)

        feature_condition_1 = ThresholdFeatureCondition(relation=">=", threshold=0)
        feature_condition_2 = ThresholdFeatureCondition(relation="<=", threshold=1)
        feature_condition_3 = ThresholdFeatureCondition(relation=">=", threshold=-1)

        graph.add_edge(feature_condition_1, feature_condition_2, index=True)
        graph.add_edge(feature_condition_1, feature_condition_3, index=False)

        for action, edge_index in zip(range(2, 4), (1, 0)):
            graph.add_edge(feature_condition_2, Action(action), index=edge_index)

        for action, edge_index in zip(range(2), (0, 1)):
            graph.add_edge(feature_condition_3, Action(action), index=edge_index)

        return graph


class F_AA_Behavior(Behavior):
    """Feature condition with mutliple actions on same index."""

    def __init__(self, name: str, any_mode: str) -> None:
        super().__init__(name, image=None)
        self.any_mode = any_mode

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self, any_mode=self.any_mode)
        feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)

        graph.add_edge(feature_condition, Action(0), index=int(True))
        graph.add_edge(feature_condition, Action(1), index=int(False))
        graph.add_edge(feature_condition, Action(2), index=int(False))

        return graph


class AF_A_Behavior(Behavior):
    """Double root with feature condition and action"""

    def __init__(self, name: str, any_mode: str) -> None:
        super().__init__(name, image=None)
        self.any_mode = any_mode

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self, any_mode=self.any_mode)

        graph.add_node(Action(0))
        feature_condition = ThresholdFeatureCondition(relation=">=", threshold=0)

        graph.add_edge(feature_condition, Action(1), index=int(True))
        graph.add_edge(feature_condition, Action(2), index=int(False))

        return graph
