# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Unit tests for the hebg.option module. """

import pytest
import pytest_check as check
from pytest_mock import MockerFixture

from hebg.behavior import Behavior


class TestOption:

    """Option"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize variables."""
        self.node = Behavior("option_name")

    def test_node_type(self):
        """should have 'option' as node_type."""
        check.equal(self.node.type, "option")

    def test_node_call(self, mocker: MockerFixture):
        """should use graph on call."""
        mocker.patch("hebg.option.Option.graph")
        self.node(None)
        check.is_true(self.node.graph.called)

    def test_build_graph(self):
        """should raise NotImplementedError when build_graph is called."""
        with pytest.raises(NotImplementedError):
            self.node.build_graph()

    def test_graph(self, mocker: MockerFixture):
        """should build graph and compute its levels if, and only if,
        the graph is not yet built.
        """
        mocker.patch("hebg.option.Option.build_graph")
        mocker.patch("hebg.option.compute_levels")
        self.node.graph
        check.is_true(self.node.build_graph.called)
        check.is_true(self.node.build_graph.called)

        mocker.patch("hebg.option.Option.build_graph")
        mocker.patch("hebg.option.compute_levels")
        self.node.graph
        check.is_false(self.node.build_graph.called)
        check.is_false(self.node.build_graph.called)
