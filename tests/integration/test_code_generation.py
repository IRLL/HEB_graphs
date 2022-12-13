import pytest_check as check

from tests.integration import FundamentalBehavior
from hebg.node import Action
from hebg.codegen import get_hebg_source


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
