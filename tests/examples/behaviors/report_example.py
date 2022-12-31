from hebg import HEBGraph, Action, FeatureCondition, Behavior


class Behavior0(Behavior):
    """Behavior 0"""

    def __init__(self) -> None:
        super().__init__("behavior 0")

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        feature = FeatureCondition("feature 0", complexity=1)
        graph.add_edge(feature, Action(0, complexity=1), index=False)
        graph.add_edge(feature, Action(1, complexity=1), index=True)
        return graph


class Behavior1(Behavior):
    """Behavior 1"""

    def __init__(self) -> None:
        super().__init__("behavior 1")

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        feature_1 = FeatureCondition("feature 1", complexity=1)
        feature_2 = FeatureCondition("feature 2", complexity=1)
        graph.add_edge(feature_1, Behavior0(), index=False)
        graph.add_edge(feature_1, feature_2, index=True)
        graph.add_edge(feature_2, Action(0, complexity=1), index=False)
        graph.add_edge(feature_2, Action(2, complexity=1), index=True)
        return graph


class Behavior2(Behavior):
    """Behavior 2"""

    def __init__(self) -> None:
        super().__init__("behavior 2")

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        feature_3 = FeatureCondition("feature 3", complexity=1)
        feature_4 = FeatureCondition("feature 4", complexity=1)
        feature_5 = FeatureCondition("feature 5", complexity=1)
        graph.add_edge(feature_3, feature_4, index=False)
        graph.add_edge(feature_3, feature_5, index=True)
        graph.add_edge(feature_4, Action(0, complexity=1), index=False)
        graph.add_edge(feature_4, Behavior1(), index=True)
        graph.add_edge(feature_5, Behavior1(), index=False)
        graph.add_edge(feature_5, Behavior0(), index=True)
        return graph
