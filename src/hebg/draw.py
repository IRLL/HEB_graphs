import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerPatch
from networkx import draw_networkx_edges
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module

from hebg.graph import draw_networkx_nodes_images
from hebg.layouts import staircase_layout
from hebg.unrolling import group_behaviors_points

if TYPE_CHECKING:
    from hebg.heb_graph import HEBGraph
    from hebg.node import Node


def draw_hebgraph(
    graph: "HEBGraph",
    ax: Axes,
    pos: Optional[Dict["Node", Tuple[float, float]]] = None,
    fontcolor: str = "black",
    draw_hulls: bool = False,
    show_all_hulls: bool = False,
) -> Tuple["Axes", Dict["Node", Tuple[float, float]]]:
    if len(list(graph.nodes())) == 0:
        return

    ax.set_title(graph.behavior.name, fontdict={"color": "orange"})
    plt.setp(ax.spines.values(), color="orange")

    if pos is None:
        pos = staircase_layout(graph)
    draw_networkx_nodes_images(graph, pos, ax=ax, img_zoom=0.5)

    draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        arrowsize=20,
        arrowstyle="-|>",
        min_source_margin=0,
        min_target_margin=10,
        node_shape="s",
        node_size=1500,
        edge_color=[color for _, _, color in graph.edges(data="color")],
    )

    legend = draw_graph_legend(graph, ax)
    plt.setp(legend.get_texts(), color=fontcolor)

    if draw_hulls:
        group_and_draw_hulls(graph, pos, ax, show_all_hulls=show_all_hulls)


def draw_graph_legend(graph: "HEBGraph", ax: Axes) -> Legend:
    used_node_types = [node_type for _, node_type in graph.nodes(data="type")]
    legend_patches = [
        mpatches.Patch(facecolor="none", edgecolor=color, label=node_type.capitalize())
        for node_type, color in graph.NODES_COLORS.items()
        if node_type in used_node_types and node_type in graph.NODES_COLORS
    ]
    used_edge_indexes = [index for _, _, index in graph.edges(data="index")]
    legend_arrows = [
        mpatches.FancyArrow(
            *(0, 0, 1, 0),
            facecolor=color,
            edgecolor="none",
            label=str(index) if index > 1 else f"{str(bool(index))} ({index})",
        )
        for index, color in graph.EDGES_COLORS.items()
        if index in used_edge_indexes and index in graph.EDGES_COLORS
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

    return legend


def group_and_draw_hulls(graph: "HEBGraph", pos, ax: Axes, show_all_hulls: bool):
    grouped_points = group_behaviors_points(pos, graph)
    if not show_all_hulls:
        key_count = {key[-1]: 0 for key in grouped_points}
        for key in grouped_points:
            key_count[key[-1]] += 1
        grouped_points = {
            key: points
            for key, points in grouped_points.items()
            if key_count[key[-1]] > 1 and (len(key) == 1 or key[-1] != key[-2])
        }

    for group_key, points in grouped_points.items():
        stretch = 0.5 - 0.05 * (len(group_key) - 1)
        if len(points) >= 3:
            draw_convex_hull(points, ax, stretch=stretch, lw=3, color="orange")


def draw_convex_hull(points, ax: "Axes", stretch=0.3, n_points=30, **kwargs):
    points = np.array(points)
    convh = ConvexHull(points)  # Get the first convexHull (speeds up the next process)
    points = buffer_points(points[convh.vertices], stretch=stretch, samples=n_points)

    hull = ConvexHull(points)
    hull_cycle = np.concatenate((hull.vertices, hull.vertices[:1]))
    ax.plot(points[hull_cycle, 0], points[hull_cycle, 1], **kwargs)


def buffer_points(inside_points, stretch, samples):
    new_points = []
    for point in inside_points:
        new_points += points_in_circum(point, stretch, samples)
    new_points = np.array(new_points)
    hull = ConvexHull(new_points)
    return new_points[hull.vertices]


def points_in_circum(points, radius, samples=100):
    return [
        (
            points[0] + math.cos(2 * math.pi / samples * x) * radius,
            points[1] + math.sin(2 * math.pi / samples * x) * radius,
        )
        for x in range(0, samples + 1)
    ]
