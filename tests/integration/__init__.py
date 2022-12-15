# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Integration tests for the heb_graph package. """

from typing import Dict, Union
from enum import Enum

from hebg.node import Action, EmptyNode, FeatureCondition
from hebg.behavior import Behavior
from hebg.heb_graph import HEBGraph


class ThresholdFeatureCondition(FeatureCondition):

    """Threshold-based feature condition for scalar feature."""

    class Relation(Enum):
        GREATER_OR_EQUAL_TO = ">="
        LESSER_OR_EQUAL_TO = "<="
        GREATER_THAN = ">"
        LESSER_THAN = "<"

    def __init__(
        self, relation: Union[Relation, str] = ">=", threshold: float = 0
    ) -> None:
        self.relation = relation
        self.threshold = threshold
        self._relation = self.Relation(relation)
        threshold_str = str(threshold).replace("-", "n")
        name = f"{self._relation.name.capitalize()} {threshold_str} ?"
        super().__init__(name=name, image=None)

    def __call__(self, observation: float) -> int:
        conditions = {
            self.Relation.GREATER_OR_EQUAL_TO: int(observation >= self.threshold),
            self.Relation.LESSER_OR_EQUAL_TO: int(observation <= self.threshold),
            self.Relation.GREATER_THAN: int(observation > self.threshold),
            self.Relation.LESSER_THAN: int(observation < self.threshold),
        }
        if self._relation in conditions:
            return conditions[self._relation]


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


class E_A_Behavior(Behavior):

    """Empty behavior"""

    def __init__(self, name: str, action: Action) -> None:
        super().__init__(name)
        self.action = action

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        graph.add_edge(EmptyNode("empty"), self.action)
        return graph
