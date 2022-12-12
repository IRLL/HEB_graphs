# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>
# pylint: disable=arguments-differ

""" Module containing the HEBGraph base class. """

from __future__ import annotations

from copy import copy, deepcopy
from typing import Any, Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.legend_handler import HandlerPatch
from networkx import DiGraph, draw_networkx_edges, relabel_nodes

from hebg.draw_utils import draw_convex_hull
from hebg.graph import draw_networkx_nodes_images, get_roots
from hebg.layouts import staircase_layout
from hebg.node import Node
from hebg.behavior import Behavior

OPTIONS_SEPARATOR = "\n>"


class HEBGraph(DiGraph):

    """Base class for Hierchical Explanation of Behavior as Graphs.

    An HEBGraph is a DiGraph, and as such stores nodes and directed edges with
    optional data, or attributes.

    But nodes of an HEBGraph are not arbitrary.
    Leaf nodes can either be an Action or a Behavior.
    Other nodes can either be a FeatureCondition or an EmptyNode.

    An HEBGraph determines the behavior of an option, it can be called with an observation
    to return the action given by this option.

    An HEBGraph edges are directed and indexed,
    this indexing for path making when calling the graph.

    As in a DiGraph loops are allowed but multiple (parallel) edges are not.

    Args:
        option: The Option object from which this graph is built.
        all_options: A dictionary of Option, this can be used to avoid cirular definitions using
            the option names as anchor instead of the Option object itself.
        any_mode: How to choose path, when multiple path are valid.
        incoming_graph_data: Additional data to include in the graph.

    """

    NODES_COLORS = {"feature_condition": "blue", "action": "red", "option": "orange"}
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
        option: Behavior,
        all_options: Dict[str, Behavior] = None,
        incoming_graph_data=None,
        any_mode: str = "first",
        **attr,
    ):
        self.option = option
        self.all_options = all_options if all_options is not None else {}

        self._unrolled_graph = None

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

    def _get_any_action(self, nodes: List[Node], observation, options_in_search: list):
        actions = []
        for node in nodes:
            action = self._get_action(node, observation, options_in_search)
            if action is None:
                return None
            actions.append(action)
        actions = [action for action in actions if action != "Impossible"]
        if len(actions) == 0:
            return "Impossible"
        if self.any_mode == "first":
            return actions[0]
        if self.any_mode == "last":
            return actions[-1]
        if self.any_mode == "random":
            return np.random.choice(actions)

    @property
    def unrolled_graph(self) -> HEBGraph:
        """Access to the unrolled option graph.

        The unrolled option graph as the same behavior but every option node is recursively replaced
        by it's option graph if it can be computed.

        Only build's the graph the first time called for efficiency.

        Returns:
            This HEBGraph's unrolled HEBGraph.

        """
        if self._unrolled_graph is None:
            self._unrolled_graph = self.build_unrolled_graph()
        return self._unrolled_graph

    def build_unrolled_graph(self) -> HEBGraph:
        """Build the the unrolled option graph.

        The unrolled option graph as the same behavior but every option node is recursively replaced
        by it's option graph if it can be computed.

        Returns:
            This HEBGraph's unrolled HEBGraph.

        """

        def add_prefix(graph, prefix: str) -> None:
            """Rename graph to obtain disjoint node labels."""
            if prefix is None:
                return graph

            def rename(x: Node):
                x_new = copy(x)
                x_new.name = prefix + x.name
                return x_new

            return relabel_nodes(graph, rename, copy=False)

        unrolled_graph: HEBGraph = copy(self)
        for node in unrolled_graph.nodes():
            node: Node = node  # Add typechecking
            node_graph: HEBGraph = None

            if node.type == "option":
                node: Behavior = node  # Add typechecking
                try:
                    try:
                        node_graph = node.graph.unrolled_graph
                    except NotImplementedError:
                        node_graph = self.all_options[str(node)].graph.unrolled_graph

                    # Relabel graph nodes to obtain disjoint node labels (if more that one node).
                    if len(node_graph.nodes()) > 1:
                        add_prefix(node_graph, str(node) + OPTIONS_SEPARATOR)

                    # Replace the option node by the unrolled option's graph
                    unrolled_graph = compose_heb_graphs(unrolled_graph, node_graph)
                    for edge_u, _, data in unrolled_graph.in_edges(node, data=True):
                        for root in node_graph.roots:
                            unrolled_graph.add_edge(edge_u, root, **data)
                    for _, edge_v, data in unrolled_graph.out_edges(node):
                        for root in node_graph.roots:
                            unrolled_graph.add_edge(root, edge_v, **data)

                    unrolled_graph.remove_node(node)
                except NotImplementedError:
                    pass

        return unrolled_graph

    def _get_action(self, node: Node, observation, options_in_search: list):
        # Option
        if node.type == "option":
            if str(node) in options_in_search:
                return "Impossible"
            try:
                return node(observation, options_in_search)
            except NotImplementedError:
                # Search in all_options, used to avoid cycling definitions
                return self.all_options[str(node)](observation, options_in_search)
        # Action
        if node.type == "action":
            return node(observation)
        # Feature Condition
        if node.type == "feature_condition":
            next_edge_index = int(node(observation))
            succs = self.successors(node)
            next_nodes = []
            for next_node in succs:
                if int(self.edges[node, next_node]["index"]) == next_edge_index:
                    next_nodes.append(next_node)
            if len(next_nodes) == 0:
                raise ValueError(
                    f"FeatureCondition {node} returned index {next_edge_index}"
                    f" but {next_edge_index} was not found as an edge index"
                )
            return self._get_any_action(next_nodes, observation, options_in_search)
        # Empty
        if node.type == "empty":
            next_node = self.successors(node).__next__()
            return self._get_action(next_node, observation, options_in_search)
        raise ValueError(f"Unknowed value {node.type} for node.type with node: {node}.")

    def __call__(self, observation, options_in_search=None) -> Any:
        options_in_search = (
            [] if options_in_search is None else deepcopy(options_in_search)
        )
        options_in_search.append(self.option.name)
        return self._get_any_action(self.roots, observation, options_in_search)

    @property
    def roots(self) -> List[Node]:
        """Roots of the option graph (nodes without predecessors)."""
        return get_roots(self)

    def draw(self, ax: Axes, **kwargs) -> Tuple[Axes, Dict[Node, Tuple[float, float]]]:
        """Draw the HEBGraph on the given Axis.

        Args:
            ax: The matplotlib ax to draw on.

        Kwargs:
            fontcolor: Font color to use for all texts.

        Returns:
            The resulting matplotlib Axis drawn on and a dictionary of each node position.

        """
        fontcolor = kwargs.get("fontcolor", "black")
        pos = kwargs.get("pos")
        if len(list(self.nodes())) > 0:
            if pos is None:
                pos = staircase_layout(self)
            draw_networkx_nodes_images(self, pos, ax=ax, img_zoom=0.5)

            draw_networkx_edges(
                self,
                pos,
                ax=ax,
                arrowsize=20,
                arrowstyle="-|>",
                min_source_margin=0,
                min_target_margin=10,
                node_shape="s",
                node_size=1500,
                edge_color=[color for _, _, color in self.edges(data="color")],
            )

            used_node_types = [node_type for _, node_type in self.nodes(data="type")]
            legend_patches = [
                mpatches.Patch(
                    facecolor="none", edgecolor=color, label=node_type.capitalize()
                )
                for node_type, color in self.NODES_COLORS.items()
                if node_type in used_node_types and node_type in self.NODES_COLORS
            ]
            used_edge_indexes = [index for _, _, index in self.edges(data="index")]
            legend_arrows = [
                mpatches.FancyArrow(
                    *(0, 0, 1, 0),
                    facecolor=color,
                    edgecolor="none",
                    label=str(index) if index > 1 else f"{str(bool(index))} ({index})",
                )
                for index, color in self.EDGES_COLORS.items()
                if index in used_edge_indexes and index in self.EDGES_COLORS
            ]

            # Draw the legend
            legend = ax.legend(
                fancybox=True,
                framealpha=0,
                fontsize="x-large",
                loc="upper right",
                handles=legend_patches + legend_arrows,
                handler_map={
                    # Patch arrows with fancy arrows in legend
                    mpatches.FancyArrow: HandlerPatch(
                        patch_func=lambda width, height, **kwargs: mpatches.FancyArrow(
                            *(0, 0.5 * height, width, 0),
                            width=0.2 * height,
                            length_includes_head=True,
                            head_width=height,
                            overhang=0.5,
                        )
                    ),
                },
            )
            plt.setp(legend.get_texts(), color=fontcolor)

            if kwargs.get("draw_options_hulls", False):
                grouped_points = group_options_points(pos, self)
                if not kwargs.get("show_all_hulls", False):
                    key_count = {key[-1]: 0 for key in grouped_points}
                    for key in grouped_points:
                        key_count[key[-1]] += 1
                    grouped_points = {
                        key: points
                        for key, points in grouped_points.items()
                        if key_count[key[-1]] > 1
                        and (len(key) == 1 or key[-1] != key[-2])
                    }

                for group_key, points in grouped_points.items():
                    stretch = 0.5 - 0.05 * (len(group_key) - 1)
                    if len(points) >= 3:
                        draw_convex_hull(
                            points, ax, stretch=stretch, lw=3, color="orange"
                        )

        return ax, pos


def compose_heb_graphs(G: HEBGraph, H: HEBGraph):
    """Returns a new graph of G composed with H.

    Composition is the simple union of the node sets and edge sets.
    The node sets of G and H do not need to be disjoint.

    Args:
        G, H : Option graphs to compose.

    Returns:
        R: A new option graph  with the same type as G

    """
    R = HEBGraph(G.option, all_options=G.all_options, any_mode=G.any_mode)
    # add graph attributes, H attributes take precedent over G attributes
    R.graph.update(G.graph)
    R.graph.update(H.graph)

    R.add_nodes_from(G.nodes(data=True))
    R.add_nodes_from(H.nodes(data=True))

    if G.is_multigraph():
        R.add_edges_from(G.edges(keys=True, data=True))
    else:
        R.add_edges_from(G.edges(data=True))
    if H.is_multigraph():
        R.add_edges_from(H.edges(keys=True, data=True))
    else:
        R.add_edges_from(H.edges(data=True))
    return R


def group_options_points(pos: Dict[Node, tuple], graph: HEBGraph) -> Dict[tuple, list]:
    """Group nodes positions of an HEBGraph in options.

    Args:
        pos (Dict[Node, tuple]): Positions of nodes.
        graph (HEBGraph): Graph.

    Returns:
        Dict[tuple, list]: A dictionary of nodes grouped by their option hierarchy.
    """
    points_grouped_by_option: Dict[tuple, list] = {}
    for node in graph.nodes():
        groups = str(node).split(OPTIONS_SEPARATOR)
        if len(groups) > 1:
            for i in range(len(groups[:-1])):
                key = tuple(groups[: -1 - i])
                point = pos[node]
                try:
                    points_grouped_by_option[key].append(point)
                except KeyError:
                    points_grouped_by_option[key] = [point]
    return points_grouped_by_option
