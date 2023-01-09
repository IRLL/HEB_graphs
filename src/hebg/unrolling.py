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

    def add_prefix_to_graph(graph: "HEBGraph", prefix: str) -> None:
        """Rename graph to obtain disjoint node labels."""
        if prefix is None:
            return graph

        def rename(x: Node):
            x_new = copy(x)
            x_new.name = prefix + x.name
            return x_new

        return relabel_nodes(graph, rename, copy=False)

    unrolled_graph: "HEBGraph" = copy(graph)
    for node in unrolled_graph.nodes():
        node: Node = node  # Add typechecking
        node_graph: "HEBGraph" = None

        if node.type == "behavior":
            node: Behavior = node  # Add typechecking
            try:
                try:
                    node_graph = node.graph.unrolled_graph
                except NotImplementedError:
                    if str(node) in graph.all_behaviors:
                        noderef_graph = graph.all_behaviors[str(node)].graph
                        node_graph = unroll_graph(noderef_graph, add_prefix=add_prefix)
                    else:
                        # If we don't find any reference, we keep it as is.
                        continue

                # Relabel graph nodes to obtain disjoint node labels (if more that one node).
                if add_prefix and len(node_graph.nodes()) > 1:
                    add_prefix_to_graph(node_graph, str(node) + BEHAVIOR_SEPARATOR)

                # Replace the behavior node by the unrolled behavior's graph
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

    compute_levels(unrolled_graph)
    return unrolled_graph


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


def compose_heb_graphs(G: "HEBGraph", H: "HEBGraph"):
    """Returns a new graph of G composed with H.

    Composition is the simple union of the node sets and edge sets.
    The node sets of G and H do not need to be disjoint.

    Args:
        G, H : HEBGraphs to compose.

    Returns:
        R: A new HEBGraph with the same type as G.

    """
    R = G.__class__(G.behavior, all_behaviors=G.all_behaviors, any_mode=G.any_mode)
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
