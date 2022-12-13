import pytest_check as check

from hebg.node import Action, FeatureCondition
from hebg.behavior import Behavior
from hebg.heb_graph import HEBGraph

from hebg.codegen import get_hebg_source


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


def test_a_graph_codegen():
    """(A) Fundamental behaviors (single action) should generate __call__ content only."""
    action = Action(42)
    a_graph = FundamentalBehavior(action).graph
    source_code = get_hebg_source(a_graph)
    expected_source_code = "\n".join(
        (
            "class Action42Behavior:",
            "    def __init__(self):",
            "        self.action_42 = Action(42)",
            "    def __call__(self, observation):",
            "        return self.action_42(observation)",
        )
    )
    check.equal(source_code, expected_source_code)
