import pytest
import pytest_check as check

from hebg.unrolling import unroll_graph

from tests.examples.behaviors.loop import build_looping_behaviors

import matplotlib.pyplot as plt


class TestLoop:
    """Tests for the loop example"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.behaviors = build_looping_behaviors()

    def test_unrolling(self):
        draw = False
        for behavior in self.behaviors:
            unrolled_graph = unroll_graph(behavior.graph)
            if draw:
                fig, ax = plt.subplots()
                plt.setp(ax.spines.values(), color="orange")
                unrolled_graph.draw(ax)
                plt.title(behavior.name, fontdict={"color": "orange"})
                plt.show()
