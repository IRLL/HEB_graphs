"""Module to unroll HEBGraph.

Unrolling means expanding each sub-behavior node as it's own graph in the global HEBGraph.
Behaviors that do not have a graph (Unexplainable behaviors) should stay as is in the graph.

"""
from copy import copy
from typing import TYPE_CHECKING, Dict

from networkx import relabel_nodes

from hebg.node import Node
from hebg.behavior import Behavior
from hebg.graph import compute_levels

BEHAVIOR_SEPARATOR = ">"

if TYPE_CHECKING:
    from hebg.heb_graph import HEBGraph


def unroll_graph(graph: "HEBGraph", add_prefix=True) -> "HEBGraph":
    """Build the the unrolled HEBGraph.

    The HEBGraph as the same behavior but every behavior node is recursively replaced
    by it's own HEBGraph if it can be computed.

    Returns:
        This HEBGraph's unrolled HEBGraph.

    """

    unrolled_graph: "HEBGraph" = copy(graph)
    for node in unrolled_graph.nodes():
        if not isinstance(node, Behavior):
            continue
        unrolled_graph = unroll_behavior(unrolled_graph, node, add_prefix)

    compute_levels(unrolled_graph)
    return unrolled_graph


def unroll_behavior(
    graph: "HEBGraph", behavior: Behavior, add_prefix: bool
) -> "HEBGraph":
    """Unroll a behavior node in a given HEBGraph

    Args:
        graph (HEBGraph): HEBGraph to unroll the behavior in.
        behavior (Behavior): Behavior node to unroll, must be in the given graph.
        add_prefix (bool): If True, adds a name prefix to keep nodes different.

    Returns:
        HEBGraph: Initial graph with unrolled behavior.
    """
    # Look for name reference.
    if str(behavior) in graph.all_behaviors:
        behavior = graph.all_behaviors[behavior.name]
    try:
        node_graph = unroll_graph(behavior.graph, add_prefix=add_prefix)
    except NotImplementedError:
        # If we cannot unroll, we keep it as is
        return graph

    # Relabel graph nodes to obtain disjoint node labels (if more that one node).
    if add_prefix and len(node_graph.nodes()) > 1:
        add_prefix_to_graph(node_graph, behavior.name + BEHAVIOR_SEPARATOR)

    # Replace the behavior node by the unrolled behavior's graph
    graph = compose_heb_graphs(graph, node_graph)
    for edge_u, _, data in graph.in_edges(behavior, data=True):
        for root in node_graph.roots:
            graph.add_edge(edge_u, root, **data)
    for _, edge_v, data in graph.out_edges(behavior):
        for root in node_graph.roots:
            graph.add_edge(root, edge_v, **data)

    graph.remove_node(behavior)
    return graph


def add_prefix_to_graph(graph: "HEBGraph", prefix: str) -> None:
    """Rename graph to obtain disjoint node labels."""
    if prefix is None:
        return graph

    def rename(node: Node):
        new_node = copy(node)
        new_node.name = prefix + node.name
        return new_node

    return relabel_nodes(graph, rename, copy=False)


def group_behaviors_points(
    pos: Dict[Node, tuple],
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
        graph_of_reference.behavior,
        all_behaviors=graph_of_reference.all_behaviors,
        any_mode=graph_of_reference.any_mode,
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
