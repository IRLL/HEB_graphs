from typing import Any, List

from hebg import HEBGraph, Action, FeatureCondition, Behavior


class HasItem(FeatureCondition):
    def __init__(self, item_name: str) -> None:
        self.item_name = item_name
        super().__init__(name=f"Has {item_name} ?")

    def __call__(self, observation: Any) -> int:
        return self.item_name in observation


class GatherWood(Behavior):
    """Gather wood"""

    def __init__(self) -> None:
        """Gather wood"""
        super().__init__("Gather wood")

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        has_axe = HasItem("axe")
        graph.add_edge(has_axe, Action("Punch tree", cost=2.0), index=False)
        graph.add_edge(has_axe, Behavior("Get new axe"), index=False)
        graph.add_edge(has_axe, Action("Use axe on tree"), index=True)
        return graph


class GetNewAxe(Behavior):
    """Get new axe with wood"""

    def __init__(self) -> None:
        """Get new axe with wood"""
        super().__init__("Get new axe")

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        has_wood = HasItem("wood")
        graph.add_edge(has_wood, Behavior("Gather wood"), index=False)
        graph.add_edge(
            has_wood, Action("Summon axe out of thin air", cost=10.0), index=False
        )
        graph.add_edge(has_wood, Action("Craft axe"), index=True)
        return graph


def build_looping_behaviors() -> List[Behavior]:
    behaviors: List[Behavior] = [GatherWood(), GetNewAxe()]
    all_behaviors = {behavior.name: behavior for behavior in behaviors}
    for behavior in behaviors:
        behavior.graph.all_behaviors = all_behaviors
    return behaviors
