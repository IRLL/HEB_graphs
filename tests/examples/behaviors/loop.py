from typing import List

from hebg import HEBGraph, Action, FeatureCondition, Behavior


class GatherWood(Behavior):
    """Gather wood"""

    def __init__(self) -> None:
        """Gather wood"""
        super().__init__("Gather wood")

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        feature = FeatureCondition("Has an axe")
        graph.add_edge(feature, Action("Punch tree"), index=False)
        graph.add_edge(feature, Behavior("Get new axe"), index=False)
        graph.add_edge(feature, Action("Use axe on tree"), index=True)
        return graph


class GetNewAxe(Behavior):
    """Get new axe with wood"""

    def __init__(self) -> None:
        """Get new axe with wood"""
        super().__init__("Get new axe")

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        feature = FeatureCondition("Has wood")
        graph.add_edge(feature, Behavior("Gather wood"), index=False)
        graph.add_edge(feature, Action("Craft axe"), index=True)
        return graph


def build_looping_behaviors() -> List[Behavior]:
    behaviors: List[Behavior] = [GatherWood(), GetNewAxe()]
    all_behaviors = {behavior.name: behavior for behavior in behaviors}
    for behavior in behaviors:
        behavior.graph.all_behaviors = all_behaviors
    return behaviors
