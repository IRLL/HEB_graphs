# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>
# pylint: disable=protected-access

""" Additional utility functions for networkx graphs. """

from networkx import DiGraph
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.axes import Axes

def get_roots(graph:DiGraph):
    """ Finds roots in a DiGraph.

    Args:
        graph: A networkx DiGraph.

    Returns:
        List of root nodes.

    """
    roots = []
    for node in graph.nodes():
        if len(list(graph.predecessors(node))) == 0:
            roots.append(node)
    return roots

def compute_levels(graph:DiGraph, verbose=0):
    """ Compute the hierachical levels of all DiGraph nodes.

    Adds the attribute 'level' to each node in the given graph.
    Adds the attribute 'nodes_by_level' to the given graph.
    Adds the attribute 'depth' to the given graph.
    The DiGraph must not have any circuit.

    Args:
        graph: A networkx DiGraph.

    Returns:
        Dictionary of nodes by level.

    """

    def _compute_level_dependencies(graph:DiGraph, node, predecessors):
        pred_level_by_index = {}
        incomplete = False
        for pred in predecessors:
            index = graph.edges[pred, node]['index']
            try:
                pred_level = graph.nodes[pred]['level']
                if index in pred_level_by_index:
                    pred_level_by_index[index].append(pred_level)
                else:
                    pred_level_by_index[index] = [pred_level]
            except KeyError:
                incomplete = True
        min_level_by_index = [min(level_list) for _, level_list in pred_level_by_index.items()]
        return 1 + max(min_level_by_index), incomplete

    all_nodes_have_level = False
    while not all_nodes_have_level:
        all_nodes_have_level = True
        for node in graph.nodes():
            predecessors = list(graph.predecessors(node))

            if len(predecessors) == 0:
                level = 0
            else:
                level, incomplete = _compute_level_dependencies(graph, node, predecessors)
                if incomplete:
                    all_nodes_have_level = False
            if 'level' in graph.nodes[node] and graph.nodes[node]['level'] != level:
                all_nodes_have_level = False

            graph.nodes[node]['level'] = level

    nodes_by_level = {}
    for node, level in graph.nodes(data='level'):
        try:
            nodes_by_level[level].append(node)
        except KeyError:
            nodes_by_level[level] = [node]

    graph.graph['nodes_by_level'] = nodes_by_level
    graph.graph['depth'] = max(level for level in nodes_by_level)
    return nodes_by_level

def compute_edges_color(graph:DiGraph):
    """ Compute the edges colors of a leveled graph for readability.

    Adds the attribute 'color' and 'linestyle' to each edge in the given graph.
    Nodes with a lot of successors will have more transparent edges.
    Edges going from high to low level will be dashed.
    Requires nodes to have a 'level' attribute.

    Args:
        graph: A networkx DiGraph.

    """
    alphas = [1, 1, 0.9, 0.8, 0.7, 0.5, 0.4, 0.3]
    for node in graph.nodes():
        successors = list(graph.successors(node))
        for succ in successors:
            alpha = 0.2
            if graph.nodes[node]['level'] < graph.nodes[succ]['level']:
                if len(successors) < len(alphas):
                    alpha = alphas[len(successors)-1]
            else:
                graph.edges[node, succ]['linestyle'] = "dashed"
            if isinstance(graph.edges[node, succ]['color'], list):
                graph.edges[node, succ]['color'][3] = alpha

def draw_networkx_nodes_images(graph:DiGraph, pos, ax:Axes, img_zoom:float=1):
    """ Draw nodes images of a networkx DiGraph on a given matplotlib ax.

    Args:
        graph: A networkx DiGraph.
        pos: Layout positions of the graph.
        ax: A matplotlib Axes.
        img_zoom (Optional): Zoom to apply to images.

    """
    for n in graph:
        img = graph.nodes[n]['image']
        color = graph.nodes[n]['color']
        if img is not None:
            min_dim = min(img.shape[:2])
            min_ax_shape = min(ax._position.width, ax._position.height)
            zoom = 100 * img_zoom * min_ax_shape / min_dim
            imagebox = OffsetImage(img, zoom=zoom)
            imagebox = AnnotationBbox(
                imagebox, pos[n],
                frameon=True, box_alignment=(0.5, 0.5)
            )

            imagebox.patch.set_facecolor('None')
            imagebox.patch.set_edgecolor(color)
            imagebox.patch.set_linewidth(3)
            imagebox.patch.set_boxstyle("round", pad=0.15)
            ax.add_artist(imagebox)
