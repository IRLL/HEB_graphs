# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>
# pylint: disable=protected-access, unused-argument, missing-function-docstring

""" Unit tests for the hebg.behavior module. """

from copy import deepcopy
from networkx.algorithms.shortest_paths.unweighted import predecessor

import pytest
import pytest_check as check
from pytest_mock import MockerFixture

from hebg.heb_graph import HEBGraph, DiGraph, Behavior


class TestHEBGraph:

    """HEBGraph"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize variables."""
        self.behavior = Behavior("base_behavior_name")
        self.heb_graph = HEBGraph(self.behavior)

    def test_init(self):
        """should instanciate correctly."""
        graph = self.heb_graph
        check.equal(graph.behavior, self.behavior)
        check.equal(graph.all_behaviors, {})
        check.equal(graph.any_mode, "first")
        check.is_true(isinstance(graph, DiGraph))

    def test_add_node(self, mocker: MockerFixture):
        """should add Node to the graph correctly."""
        mocker.patch("hebg.heb_graph.DiGraph.add_node")

        class DummyNode:
            """DummyNode"""

            name = "node_name"
            type = "node_type"
            image = "node_image"

            def __str__(self) -> str:
                return self.name

        node = DummyNode()
        expected_kwargs = {"type": "node_type", "image": "node_image", "color": None}
        expected_args = (node,)

        self.heb_graph.add_node(node)
        args, kwargs = DiGraph.add_node.call_args
        check.equal(kwargs, expected_kwargs)
        check.equal(args, expected_args)

    def test_add_edge_only(self, mocker: MockerFixture):
        """should add edges to the graph correctly if nodes already exists."""
        mocker.patch("hebg.heb_graph.DiGraph.add_edge")

        class DummyNode:
            """DummyNode"""

            type = "node_type"
            image = "node_image"

            def __init__(self, i) -> None:
                self.name = f"node_name_{i}"

            def __str__(self) -> str:
                return self.name

        node_0, node_1 = DummyNode(0), DummyNode(1)
        self.heb_graph.add_node(node_0)
        self.heb_graph.add_node(node_1)

        mocker.patch("hebg.heb_graph.DiGraph.add_node")
        expected_kwargs = {"index": 42, "color": "black"}
        expected_args = (node_0, node_1)

        self.heb_graph.add_edge(node_0, node_1, index=42)
        args, kwargs = DiGraph.add_edge.call_args
        check.equal(kwargs, expected_kwargs)
        check.equal(args, expected_args)
        check.is_false(DiGraph.add_node.called)

    def test_add_edge_and_nodes(self, mocker: MockerFixture):
        """should add edges and nodes to the graph correctly if nodes are not in the graph yet."""
        mocker.patch("hebg.heb_graph.DiGraph.add_edge")
        mocker.patch("hebg.heb_graph.DiGraph.add_node")

        class DummyNode:
            """DummyNode"""

            type = "node_type"
            image = "node_image"

            def __init__(self, i) -> None:
                self.name = f"node_name_{i}"

            def __str__(self) -> str:
                return self.name

        node_0, node_1 = DummyNode(0), DummyNode(1)

        expected_nodes_args = ((node_0,), (node_1,))
        expected_edge_kwargs = {"index": 42, "color": "black"}
        expected_edge_args = (node_0, node_1)

        self.heb_graph.add_edge(node_0, node_1, index=42)
        args, kwargs = DiGraph.add_edge.call_args
        check.equal(kwargs, expected_edge_kwargs)
        check.equal(args, expected_edge_args)

        for i, (args, _) in enumerate(DiGraph.add_node.call_args_list):
            check.equal(args, expected_nodes_args[i])

    def test_call(self, mocker: MockerFixture):
        """should return roots action on call."""
        roots = "roots"
        observation = "obs"
        mocker.patch("hebg.heb_graph.HEBGraph._get_any_action")
        mocker.patch("hebg.heb_graph.HEBGraph.roots", roots)
        self.heb_graph(observation)
        args, _ = HEBGraph._get_any_action.call_args
        check.equal(args[0], roots)
        check.equal(args[1], observation)
        check.equal(args[2], [self.behavior.name])

    def test_roots(self, mocker: MockerFixture):
        """should have roots as property."""

        nodes = ["A", "B", "C", "AA", "AB"]
        predecessors = {"A": [], "B": [], "C": [], "AA": ["A"], "AB": ["A", "B"]}

        mocker.patch("hebg.heb_graph.HEBGraph.nodes", lambda self: nodes)
        mocker.patch(
            "hebg.heb_graph.HEBGraph.predecessors",
            lambda self, node: predecessors[node],
        )

        check.equal(self.heb_graph.roots, ["A", "B", "C"])


class TestHEBGraphGetAnyAction:

    """HEBGraph._get_any_action"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize variables."""
        self.behavior = Behavior("behavior_name")
        self.heb_graph = HEBGraph(self.behavior)

    def test_none_in_actions(self, mocker: MockerFixture):
        """should return None if any node returns None."""

        actions = [0, "Impossible", 2, None, 3]

        _actions = deepcopy(actions)

        def mocked_get_action(*args):
            return _actions.pop(0)

        mocker.patch("hebg.heb_graph.HEBGraph._get_action", mocked_get_action)
        action = self.heb_graph._get_any_action(range(5), None, None)
        check.is_none(action)

    def test_no_actions(self, mocker: MockerFixture):
        """should return 'Impossible' if no action is possible."""

        actions = ["Impossible", "Impossible", "Impossible", "Impossible", "Impossible"]

        _actions = deepcopy(actions)

        def mocked_get_action(*args):
            return _actions.pop(0)

        mocker.patch("hebg.heb_graph.HEBGraph._get_action", mocked_get_action)
        action = self.heb_graph._get_any_action(range(5), None, None)
        check.equal(action, "Impossible")

        action = self.heb_graph._get_any_action([], None, None)
        check.equal(action, "Impossible")

    @pytest.mark.parametrize("any_mode", HEBGraph.ANY_MODES)
    def test_any_mode_(self, any_mode, mocker: MockerFixture):
        actions = [0, "Impossible", 2, 3, "Impossible"]

        _actions = deepcopy(actions)

        def mocked_get_action(*args):
            return _actions.pop(0)

        expected_actions = {"first": (0,), "last": (3,), "random": (0, 2, 3)}

        mocker.patch("hebg.heb_graph.HEBGraph._get_action", mocked_get_action)
        self.heb_graph.any_mode = any_mode
        action = self.heb_graph._get_any_action(range(5), None, None)
        check.is_in(action, expected_actions[any_mode])


class TestHEBGraphGetAction:

    """HEBGraph._get_action"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize variables."""
        self.behavior = Behavior("behavior_name")
        self.heb_graph = HEBGraph(self.behavior)

    def test_action(self):
        """should return action given by an Action node."""
        expected_action = "action_action"

        class DummyAction:
            """DummyNode"""

            type = "action"

            def __call__(self, observation):
                return expected_action

        action_node = DummyAction()
        action = self.heb_graph._get_action(action_node, None, None)
        check.equal(action, expected_action)

    def test_empty(self, mocker: MockerFixture):
        """should return next successor returns when given an Empty node."""

        expected_action = "action_action"

        class DummyAction:
            """DummyNode"""

            type = "action"

            def __call__(self, observation):
                return expected_action

        mocker.patch(
            "hebg.heb_graph.HEBGraph.successors",
            lambda self, node: iter([DummyAction()]),
        )

        class DummyEmpty:
            """DummyNode"""

            type = "empty"

        empty_node = DummyEmpty()
        action = self.heb_graph._get_action(empty_node, None, None)
        check.equal(action, expected_action)

    def test_unknowed_node_type(self):
        """should raise ValueError if node.type is unknowed."""

        class DummyNode:
            """DummyNode"""

            type = "random_type_error"

        node = DummyNode()
        with pytest.raises(ValueError):
            self.heb_graph._get_action(node, None, None)

    def test_behavior_in_search(self):
        """should return 'Impossible' if behavior is already in search to avoid cycles."""

        class DummyBehavior:
            """DummyBehavior"""

            type = "behavior"
            name = "behavior_already_in_search"

            def __str__(self) -> str:
                return self.name

        node = DummyBehavior()
        action = self.heb_graph._get_action(node, None, ["behavior_already_in_search"])
        check.equal(action, "Impossible")

    def test_behavior_call(self):
        """should return behavior's return if behavior can be called."""
        expected_action = "behavior_action"

        class DummyBehavior:
            """DummyBehavior"""

            type = "behavior"
            name = "behavior_name"

            def __str__(self) -> str:
                return self.name

            def __call__(self, *args) -> str:
                return expected_action

        node = DummyBehavior()
        action = self.heb_graph._get_action(node, None, [])
        check.equal(action, expected_action)

    def test_behavior_by_all_behaviors(self):
        """should use all_behaviors if behavior cannot be called."""
        expected_action = "behavior_action"

        class DummyTrueBehavior:
            """DummyBehavior"""

            type = "behavior"
            name = "behavior_name"

            def __str__(self) -> str:
                return self.name

            def __call__(self, *args):
                return expected_action

        class DummyBehaviorInGraph:
            """DummyBehavior"""

            type = "behavior"
            name = "behavior_name"

            def __str__(self) -> str:
                return self.name

            def __call__(self, *args):
                raise NotImplementedError

        true_node = DummyTrueBehavior()
        self.heb_graph.all_behaviors = {true_node.name: true_node}

        node_in_graph = DummyBehaviorInGraph()
        action = self.heb_graph._get_action(
            node_in_graph,
            observation=None,
            behaviors_in_search=[],
        )
        check.equal(action, expected_action)

    def test_feature_condition(self, mocker: MockerFixture):
        """should use FeatureCondition's given index to orient in graph."""

        class DummyAction:
            """DummyAction"""

            type = "action"
            name = "action_name"

            def __init__(self, i) -> None:
                self.index = i
                self.action = f"action_{i}"

            def __str__(self) -> str:
                return self.name

            def __call__(self, *args) -> str:
                return self.action

        class DummyFeatureCondition:
            """DummyFeatureCondition"""

            type = "feature_condition"
            name = "feature_condition_name"

            def __init__(self, i) -> None:
                self.fc_index = i

            def __str__(self) -> str:
                return self.name

            def __call__(self, *args) -> int:
                return self.fc_index

        actions = [DummyAction(i) for i in range(3)]
        mocker.patch(
            "hebg.heb_graph.HEBGraph.successors",
            lambda self, node: actions,
        )

        class DummyEdges:
            """DummyEdges"""

            def __getitem__(self, e):
                return {"index": e[1].index}

        mocker.patch("hebg.heb_graph.HEBGraph.edges", DummyEdges())
        mocker.patch(
            "hebg.heb_graph.HEBGraph._get_any_action",
            lambda self, next_nodes, observation, behaviors_in_search: next_nodes[0](
                observation
            ),
        )

        for fc_index in range(3):
            node = DummyFeatureCondition(fc_index)
            action = self.heb_graph._get_action(node, None, [])
            expected_action = f"action_{fc_index}"
            check.equal(action, expected_action)

    def test_feature_condition_index_value_error(self, mocker: MockerFixture):
        """should raise ValueError if FeatureCondition's given index represents no successor."""

        class DummyAction:
            """DummyAction"""

            type = "action"
            name = "action_name"

            def __init__(self, i) -> None:
                self.index = i
                self.action = f"action_{i}"

            def __str__(self) -> str:
                return self.name

            def __call__(self, *args) -> str:
                return self.action

        class DummyFeatureCondition:
            """DummyFeatureCondition"""

            type = "feature_condition"
            name = "feature_condition_name"

            def __init__(self, i) -> None:
                self.fc_index = i

            def __str__(self) -> str:
                return self.name

            def __call__(self, *args) -> int:
                return self.fc_index

        actions = [DummyAction(i) for i in range(3)]
        mocker.patch(
            "hebg.heb_graph.HEBGraph.successors",
            lambda self, node: actions,
        )

        class DummyEdges:
            """DummyEdges"""

            def __getitem__(self, e):
                return {"index": e[1].index}

        mocker.patch("hebg.heb_graph.HEBGraph.edges", DummyEdges())

        node = DummyFeatureCondition(4)
        with pytest.raises(ValueError):
            self.heb_graph._get_action(node, None, [])
