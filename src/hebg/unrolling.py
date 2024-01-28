# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2024 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""Module to unroll HEBGraph.

Unrolling means expanding each sub-behavior node as it's own graph in the global HEBGraph.
Behaviors that do not have a graph (Unexplainable behaviors) should stay as is in the graph.

"""

from copy import copy
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Union

from networkx import relabel_nodes

from hebg.behavior import Behavior

BEHAVIOR_SEPARATOR = ">"

if TYPE_CHECKING:
    from hebg import HEBGraph, Node, Action


def unroll_graph(
    graph: "HEBGraph",
    add_prefix=False,
    cut_looping_alternatives: bool = False,
) -> "HEBGraph":
    """Build the the unrolled HEBGraph.

    The HEBGraph as the same behavior but every behavior node is recursively replaced
    by it's own HEBGraph if it can be computed.

    Args:
        graph (HEBGraph): HEBGraph to unroll the behavior in.
        add_prefix (bool, optional): If True, adds a name prefix to keep nodes different.
            Defaults to False.
        cut_looping_alternatives (bool, optional): If True, cut the looping alternatives.
            Defaults to False.

    Returns:
        HEBGraph: This HEBGraph's unrolled HEBGraph.
    """
    unrolled_graph, _is_looping = _unroll_graph(
        graph,
        add_prefix=add_prefix,
        cut_looping_alternatives=cut_looping_alternatives,
    )
    return unrolled_graph


def _unroll_graph(
    graph: "HEBGraph",
    add_prefix=False,
    cut_looping_alternatives: bool = False,
    _current_alternatives: Optional[List[Union["Action", "Behavior"]]] = None,
    _unrolled_behaviors: Optional[Dict[str, Optional["HEBGraph"]]] = None,
) -> Tuple["HEBGraph", bool]:
    if _unrolled_behaviors is None:
        _unrolled_behaviors = {}
    if _current_alternatives is None:
        _current_alternatives = []

    is_looping = False
    _unrolled_behaviors[graph.behavior.name] = None

    unrolled_graph: "HEBGraph" = copy(graph)
    for node in list(unrolled_graph.nodes()):
        if not isinstance(node, Behavior):
            continue
        new_alternatives = []
        for pred, _node, data in graph.in_edges(node, data=True):
            index = data["index"]
            for _pred, alternative, alt_index in graph.out_edges(pred, data="index"):
                if index == alt_index and alternative != node:
                    new_alternatives.append(alternative)
        if new_alternatives:
            _current_alternatives = new_alternatives
        unrolled_graph, behavior_is_looping = _unroll_behavior(
            unrolled_graph,
            node,
            add_prefix,
            cut_looping_alternatives,
            _current_alternatives,
            _unrolled_behaviors,
        )
        if behavior_is_looping:
            is_looping = True

    return unrolled_graph, is_looping


def _unroll_behavior(
    graph: "HEBGraph",
    behavior: "Behavior",
    add_prefix: bool,
    cut_looping_alternatives: bool,
    _current_alternatives: List[Union["Action", "Behavior"]],
    _unrolled_behaviors: Dict[str, Optional["HEBGraph"]],
) -> Tuple["HEBGraph", bool]:
    """Unroll a behavior node in a given HEBGraph

    Args:
        graph (HEBGraph): HEBGraph to unroll the behavior in.
        behavior (Behavior): Behavior node to unroll, must be in the given graph.
        add_prefix (bool): If True, adds a name prefix to keep nodes different.
        cut_looping_alternatives (bool): If True, cut the looping alternatives.

    Returns:
        HEBGraph: Initial graph with unrolled behavior.
    """
    # Look for name reference.
    if behavior.name in graph.all_behaviors:
        behavior = graph.all_behaviors[behavior.name]

    node_graph, is_looping = _unrolled_behavior_graph(
        behavior,
        add_prefix,
        cut_looping_alternatives,
        _current_alternatives,
        _unrolled_behaviors,
    )

    if is_looping and cut_looping_alternatives:
        if not _current_alternatives:
            return graph, is_looping
        for alternative in _current_alternatives:
            for last_condition, _, data in graph.in_edges(behavior, data=True):
                graph.add_edge(last_condition, alternative, **data)
        graph.remove_node(behavior)
        return graph, False

    if node_graph is None:
        # If we cannot get the node's graph, we keep it as is.
        return graph, is_looping

    # Relabel graph nodes to obtain disjoint node labels (if more that one node).
    if add_prefix and len(node_graph.nodes()) > 1:
        _add_prefix_to_graph(node_graph, behavior.name + BEHAVIOR_SEPARATOR)

    # Replace the behavior node by the unrolled behavior's graph
    graph = compose_heb_graphs(graph, node_graph)
    for edge_u, _, data in graph.in_edges(behavior, data=True):
        for root in node_graph.roots:
            graph.add_edge(edge_u, root, **data)

    graph.remove_node(behavior)
    return graph, is_looping


def _unrolled_behavior_graph(
    behavior: "Behavior",
    add_prefix: bool,
    cut_looping_alternatives: bool,
    _current_alternatives: List[Union["Action", "Behavior"]],
    _unrolled_behaviors: Dict[str, Optional["HEBGraph"]],
) -> Optional["HEBGraph"]:
    """Get the unrolled sub-graph of a behavior.

    Args:
        behavior (Behavior): Behavior to get the unrolled graph of.
        add_prefix (bool): If True, adds a prefix in sub-hierarchies to have distinct nodes.
        cut_looping_alternatives (bool): If True, cut the looping alternatives.
        _unrolled_behaviors (Dict[str, Optional[HEBGraph]]): Dictionary of already computed
            unrolled graphs, both to save compute and prevent recursion loops.

    Returns:
        Optional[HEBGraph]: Unrolled graph of a behavior, None if it cannot be computed.
    """
    if behavior.name in _unrolled_behaviors:
        # If we have aleardy unrolled this behavior, we reuse it's graph
        is_looping = _unrolled_behaviors[behavior.name] is None
        return _unrolled_behaviors[behavior.name], is_looping

    try:
        node_graph, is_looping = _unroll_graph(
            behavior.graph,
            add_prefix=add_prefix,
            cut_looping_alternatives=cut_looping_alternatives,
            _current_alternatives=_current_alternatives,
            _unrolled_behaviors=_unrolled_behaviors,
        )
        _unrolled_behaviors[behavior.name] = node_graph
        return node_graph, is_looping
    except NotImplementedError:
        return None, False


def _add_prefix_to_graph(graph: "HEBGraph", prefix: str) -> None:
    """Rename graph to obtain disjoint node labels."""
    if prefix is None:
        return graph

    def rename(node: "Node"):
        new_node = copy(node)
        new_node.name = prefix + node.name
        return new_node

    return relabel_nodes(graph, rename, copy=False)


def group_behaviors_points(
    pos: Dict["Node", tuple],
    graph: "HEBGraph",
) -> Dict[tuple, list]:
    """Group nodes positions of an HEBGraph by sub-behavior.

    Args:
        pos (Dict[Node, tuple]): Positions of nodes.
        graph (HEBGraph): Graph.

    Returns:
        Dict[tuple, list]: A dictionary of nodes grouped by their behavior's hierarchy.
    """
    points_grouped_by_behavior: Dict[tuple, list] = {}
    for node in graph.nodes():
        groups = str(node).split(BEHAVIOR_SEPARATOR)
        if len(groups) > 1:
            for i in range(len(groups[:-1])):
                key = tuple(groups[: -1 - i])
                point = pos[node]
                try:
                    points_grouped_by_behavior[key].append(point)
                except KeyError:
                    points_grouped_by_behavior[key] = [point]
    return points_grouped_by_behavior


def compose_heb_graphs(graph_of_reference: "HEBGraph", other_graph: "HEBGraph"):
    """Returns a new_graph of graph_of_reference composed with other_graph.

    Composition is the simple union of the node sets and edge sets.
    The node sets of the graph_of_reference and other_graph do not need to be disjoint.

    Args:
        graph_of_reference, other_graph : HEBGraphs to compose.

    Returns:
        A new HEBGraph with the same type as graph_of_reference.

    """
    new_graph = graph_of_reference.__class__(
        graph_of_reference.behavior, all_behaviors=graph_of_reference.all_behaviors
    )
    # add graph attributes, H attributes take precedent over G attributes
    new_graph.graph.update(graph_of_reference.graph)
    new_graph.graph.update(other_graph.graph)

    new_graph.add_nodes_from(graph_of_reference.nodes(data=True))
    new_graph.add_nodes_from(other_graph.nodes(data=True))

    if graph_of_reference.is_multigraph():
        new_graph.add_edges_from(graph_of_reference.edges(keys=True, data=True))
    else:
        new_graph.add_edges_from(graph_of_reference.edges(data=True))
    if other_graph.is_multigraph():
        new_graph.add_edges_from(other_graph.edges(keys=True, data=True))
    else:
        new_graph.add_edges_from(other_graph.edges(data=True))
    return new_graph
