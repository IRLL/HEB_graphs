import pytest
import pytest_check as check

from tests.examples.behaviors.loop import build_looping_behaviors


class TestLoop:
    """Tests for the loop example"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.behaviors = build_looping_behaviors()

    def test_unrolling(self):
        for behavior in self.behaviors:
            behavior.graph.unrolled_graph
