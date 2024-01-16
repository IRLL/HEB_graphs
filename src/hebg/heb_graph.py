# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>
# pylint: disable=arguments-differ

""" Module containing the HEBGraph base class. """

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypeVar

import numpy as np
from matplotlib.axes import Axes
from networkx import DiGraph

from hebg.behavior import Behavior
from hebg.codegen import get_hebg_source
from hebg.draw import draw_hebgraph
from hebg.graph import get_roots, get_successors_with_index
from hebg.node import Node
from hebg.unrolling import unroll_graph


Action = TypeVar("Action")


class HEBGraph(DiGraph):

    """Base class for Hierchical Explanation of Behavior as Graphs.

    An HEBGraph is a DiGraph, and as such stores nodes and directed edges with
    optional data, or attributes.

    But nodes of an HEBGraph are not arbitrary.
    Leaf nodes can either be an Action or a Behavior.
    Other nodes can either be a FeatureCondition or an EmptyNode.

    An HEBGraph determines a behavior, it can be called with an observation
    to return the action given by this behavior.

    An HEBGraph edges are directed and indexed,
    this indexing for path making when calling the graph.

    As in a DiGraph loops are allowed but multiple (parallel) edges are not.

    Args:
        behavior: The Behavior object from which this graph is built.
        all_behaviors: A dictionary of behavior, this can be used to avoid cirular definitions using
            the behavior names as anchor instead of the behavior object itself.
        any_mode: How to choose path, when multiple path are valid.
        incoming_graph_data: Additional data to include in the graph.

    """

    NODES_COLORS = {"feature_condition": "blue", "action": "red", "behavior": "orange"}
    EDGES_COLORS = {
        0: "red",
        1: "green",
        2: "blue",
        3: "yellow",
        4: "purple",
        5: "cyan",
        6: "gray",
    }
    ANY_MODES = ("first", "last", "random")

    def __init__(
        self,
        behavior: Behavior,
        all_behaviors: Dict[str, Behavior] = None,
        incoming_graph_data=None,
        any_mode: str = "first",
        **attr,
    ):
        self.behavior = behavior
        self.all_behaviors = all_behaviors if all_behaviors is not None else {}

        self._unrolled_graph = None
        self.last_call_behaviors_stack = None

        assert any_mode in self.ANY_MODES, f"Unknowed any_mode: {any_mode}"
        self.any_mode = any_mode

        super().__init__(incoming_graph_data=incoming_graph_data, **attr)

    def add_node(self, node_for_adding: Node, **attr):
        node = node_for_adding
        color = attr.pop("color", None)
        attr.pop("type", None)
        attr.pop("image", None)
        if color is None:
            try:
                color = self.NODES_COLORS[node.type]
            except KeyError:
                color = None
        super().add_node(node, type=node.type, color=color, image=node.image, **attr)

    def add_edge(self, u_of_edge: Node, v_of_edge: Node, index: int = 1, **attr):
        for node in (u_of_edge, v_of_edge):
            if node not in self.nodes():
                self.add_node(node)

        color = attr.pop("color", None)
        if color is None:
            try:
                color = self.EDGES_COLORS[index]
            except KeyError:
                color = "black"
        super().add_edge(u_of_edge, v_of_edge, index=index, color=color, **attr)

    def _get_options(
        self,
        nodes: List[Node],
        observation,
        behaviors_in_search: list,
        last_call_behaviors_stack: Optional[list] = None,
        parent_name: Optional[str] = None,
    ) -> List[Action]:
        actions = []
        for node in nodes:
            node_action = self._get_action(
                node,
                observation,
                behaviors_in_search,
                last_call_behaviors_stack=last_call_behaviors_stack,
            )
            if node_action is None:
                return None
            actions.append(node_action)

        options = remove_duplicate_actions(
            [action for action in actions if action != "Impossible"]
        )

        if parent_name is None:
            parent_name = self.behavior.name
        if (
            (len(nodes) > 1 or self.behavior.name)
            and options
            and last_call_behaviors_stack is not None
        ):
            last_call_behaviors_stack.insert(
                0, (self.behavior.name, [n.name for n in self.roots], options)
            )

        return options

    def _choose_action(self, actions: Optional[List[Action]]) -> Action:
        if actions is None:
            return None
        if len(actions) == 0:
            return "Impossible"
        if self.any_mode == "first" or len(actions) == 1:
            return actions[0]
        if self.any_mode == "last":
            return actions[-1]
        if self.any_mode == "random":
            return np.random.choice(actions)

    def _get_any_action(
        self, nodes: List[Node], observation, behaviors_in_search: list
    ):
        return self._choose_action(
            self._get_options(nodes, observation, behaviors_in_search)
        )

    @property
    def unrolled_graph(self) -> HEBGraph:
        """Access to the unrolled behavior graph.

        The unrolled behavior graph as the same behavior but every behavior node is recursively replaced
        by it's behavior graph if it can be computed.

        Only build's the graph the first time called for efficiency.

        Returns:
            This HEBGraph's unrolled HEBGraph.

        """
        if self._unrolled_graph is None:
            self._unrolled_graph = unroll_graph(self)
        return self._unrolled_graph

    def _get_action(
        self,
        node: Node,
        observation: Any,
        behaviors_in_search: List[str],
        last_call_behaviors_stack: Optional[list] = None,
    ):
        # Behavior
        if node.type == "behavior":
            # To avoid cycling definitions
            if node.name in behaviors_in_search:
                return "Impossible"

            # Search for name reference in all_behaviors
            if node.name in self.all_behaviors:
                node = self.all_behaviors[node.name]

            return node(observation, behaviors_in_search, last_call_behaviors_stack)

        # Action
        if node.type == "action":
            return node(observation)
        # Feature Condition
        if node.type == "feature_condition":
            next_edge_index = int(node(observation))
            next_nodes = get_successors_with_index(self, node, next_edge_index)
            options = self._get_options(
                next_nodes,
                observation,
                behaviors_in_search,
                last_call_behaviors_stack=last_call_behaviors_stack,
                parent_name=node.name,
            )
            return self._choose_action(options)
        # Empty
        if node.type == "empty":
            next_node = self.successors(node).__next__()
            return self._get_action(
                next_node,
                observation,
                behaviors_in_search,
                last_call_behaviors_stack=last_call_behaviors_stack,
            )
        raise ValueError(f"Unknowed value {node.type} for node.type with node: {node}.")

    def __call__(
        self,
        observation,
        behaviors_in_search: Optional[List[str]] = None,
        last_call_behaviors_stack: Optional[list] = None,
    ) -> Any:
        if behaviors_in_search is None:
            behaviors_in_search = []
            last_call_behaviors_stack = []
        behaviors_in_search.append(self.behavior.name)
        options = self._get_options(
            self.roots,
            observation,
            behaviors_in_search,
            last_call_behaviors_stack=last_call_behaviors_stack,
        )
        self.last_call_behaviors_stack = last_call_behaviors_stack
        return self._choose_action(options)

    @property
    def roots(self) -> List[Node]:
        """Roots of the behavior graph (nodes without predecessors)."""
        return get_roots(self)

    def generate_source_code(self) -> str:
        """Generated source code of the behavior from graph."""
        return get_hebg_source(self)

    def draw(
        self, ax: "Axes", **kwargs
    ) -> Tuple["Axes", Dict[Node, Tuple[float, float]]]:
        """Draw the HEBGraph on the given Axis.

        Args:
            ax: The matplotlib ax to draw on.

        Kwargs:
            fontcolor: Font color to use for all texts.

        Returns:
            The resulting matplotlib Axis drawn on and a dictionary of each node position.

        """
        return draw_hebgraph(self, ax, **kwargs)


def remove_duplicate_actions(actions: List[Action]) -> List[Action]:
    seen = set()
    seen_add = seen.add
    return [a for a in actions if not (a in seen or seen_add(a))]
