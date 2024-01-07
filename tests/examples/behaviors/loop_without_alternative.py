from typing import List

from hebg import HEBGraph, Action, FeatureCondition, Behavior


class ReachForest(Behavior):
    """Reach forest"""

    def __init__(self) -> None:
        """Reach forest"""
        super().__init__("Reach forest")

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        is_in_other_zone = FeatureCondition("Is in other zone ?")
        graph.add_edge(is_in_other_zone, Behavior("Reach other zone"), index=False)
        graph.add_edge(is_in_other_zone, Action("> forest"), index=True)
        is_in_other_zone = FeatureCondition("Is in meadow ?")
        graph.add_edge(is_in_other_zone, Behavior("Reach meadow"), index=False)
        graph.add_edge(is_in_other_zone, Action("> forest"), index=True)
        return graph


class ReachOtherZone(Behavior):
    """Reach other zone"""

    def __init__(self) -> None:
        """Reach other zone"""
        super().__init__("Reach other zone")

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        is_in_forest = FeatureCondition("Is in forest ?")
        graph.add_edge(is_in_forest, Behavior("Reach forest"), index=False)
        graph.add_edge(is_in_forest, Action("> other zone"), index=True)
        is_in_other_zone = FeatureCondition("Is in meadow ?")
        graph.add_edge(is_in_other_zone, Behavior("Reach meadow"), index=False)
        graph.add_edge(is_in_other_zone, Action("> other zone"), index=True)
        return graph


class ReachMeadow(Behavior):
    """Reach meadow"""

    def __init__(self) -> None:
        """Reach meadow"""
        super().__init__("Reach meadow")

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        is_in_forest = FeatureCondition("Is in forest ?")
        graph.add_edge(is_in_forest, Behavior("Reach forest"), index=False)
        graph.add_edge(is_in_forest, Action("> meadow"), index=True)
        is_in_other_zone = FeatureCondition("Is in other zone ?")
        graph.add_edge(is_in_other_zone, Behavior("Reach other zone"), index=False)
        graph.add_edge(is_in_other_zone, Action("> meadow"), index=True)
        return graph


def build_looping_behaviors() -> List[Behavior]:
    behaviors: List[Behavior] = [
        ReachForest(),
        ReachOtherZone(),
        ReachMeadow(),
    ]
    all_behaviors = {behavior.name: behavior for behavior in behaviors}
    for behavior in behaviors:
        behavior.graph.all_behaviors = all_behaviors
    return behaviors
