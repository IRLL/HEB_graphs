# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""Unit tests for the hebg.node module."""

import pytest
import pytest_check as check

from hebg.node import Node, Action, FeatureCondition, EmptyNode


class TestNode:
    """Node"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize variables."""

    def test_node_type(self):
        """should have correct node_type and raise ValueError otherwise."""
        with pytest.raises(ValueError):
            Node("", "")
        for node_type in ("action", "feature_condition", "behavior", "empty"):
            node = Node("", node_type)
            check.equal(node.type, node_type)

    def test_node_name(self):
        """should have name as attribute."""
        name = "node_name"
        node = Node(name, "empty")
        check.equal(node.name, name)

    def test_node_call(self):
        """should raise NotImplementedError on call."""
        node = Node("", "empty")
        with pytest.raises(NotImplementedError):
            node(None)


class TestAction:
    """Action"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize variables."""

    def test_node_type(self):
        """should have 'action' as node_type."""
        node = Action("", "")
        check.equal(node.type, "action")

    def test_node_call(self):
        """should return Action.action when called."""
        action = "action_action"
        node = Action(action, "action_name")
        check.equal(node(None), action)
        check.equal(node.action, action)


class TestFeatureCondition:
    """FeatureCondition"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize variables."""

    def test_node_type(self):
        """should have 'feature_condition' as node_type."""
        node = FeatureCondition("")
        check.equal(node.type, "feature_condition")

    def test_node_call(self):
        """should raise NotImplementedError on call."""
        node = FeatureCondition("")
        with pytest.raises(NotImplementedError):
            node(None)


class TestEmptyNode:
    """EmptyNode"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize variables."""

    def test_node_type(self):
        """should have 'empty' as node_type."""
        node = EmptyNode("")
        check.equal(node.type, "empty")

    def test_node_call(self):
        """should return 1 when called."""
        node = EmptyNode("")
        check.equal(node(None), 1)
