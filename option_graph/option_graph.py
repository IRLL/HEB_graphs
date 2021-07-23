# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

from __future__ import annotations
from options_graphs.option_graph.node import Node

from typing import TYPE_CHECKING, Union, Any

import networkx as nx

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

from option_graph.graph import option_layout, draw_networkx_nodes_images

if TYPE_CHECKING:
    from option_graph.node import EmptyNode
    from option_graph.option import Option
    from option_graph.action import Action
    from option_graph.feature_condition import FeatureCondition


class OptionGraph(nx.DiGraph):

    NODES_COLORS = {'feature_check': 'blue', 'action': 'red',
        'option': 'orange', 'empty': 'purple'}
    EDGES_COLORS = {0:'red', 1:'green', 2:'blue', 3:'yellow',
        4:'rose', 5:'cyan', 6:'gray', 'any':'purple'}

    def __init__(self, all_options, incoming_graph_data, **attr):
        self.all_options = all_options
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)

    def add_node(self, node:Union[FeatureCondition, Action, Option]):
        if isinstance(node, FeatureCondition):
            node_type = 'feature_check'
        elif isinstance(node, Action):
            node_type = 'action'
        elif isinstance(node, Option):
            node_type = 'option'
        elif isinstance(node, EmptyNode):
            node_type = 'empty'
        else:
            raise TypeError("Node type must be one of "
                            "(FeatureCondition, Action, Option, EmptyNode)"
                            f"found {type(node)} instead")
        super().add_node(node.name, type=node_type,
            color=self.NODES_COLORS[node_type], image=node.image)

    def add_edge_condition(self, u_of_edge, v_of_edge, index:int):
        node_type = 'condition'
        self.add_edge(u_of_edge, v_of_edge, index=index,
            type=node_type, color=self.EDGES_COLORS[index])

    def add_edge_any(self, u_of_edge, v_of_edge):
        node_type = 'any'
        self.add_edge(u_of_edge, v_of_edge,
            type=node_type, color=self.EDGES_COLORS[node_type])

    def add_predecessors(self, prev_checks, node, force_any=False):
        if len(prev_checks) > 1 or (force_any and len(prev_checks) > 0):
            for pred in prev_checks:
                self.add_edge_any(pred, node)
        elif len(prev_checks) == 1:
            self.add_edge_condition(prev_checks[0], node, 0)


    def __call__(self, observation) -> Any:
        raise NotImplementedError

    def draw(self, ax:Axes, **kwargs) -> Axes:
        fontcolor = kwargs.get('fontcolor', 'black')
        if len(list(self.nodes())) > 0:
            pos = option_layout(self)
            draw_networkx_nodes_images(self, pos, ax=ax, img_zoom=0.5)

            nx.draw_networkx_edges(
                self, pos,
                ax=ax,
                arrowsize=20,
                arrowstyle="-|>",
                min_source_margin=0, min_target_margin=10,
                node_shape='s', node_size=1500,
                edge_color=[color for _, _, color in self.edges(data='color')]
            )

            legend_patches = [
                mpatches.Patch(
                    facecolor='none',
                    edgecolor=self.NODES_COLORS[node_type],
                    label=node_type.capitalize()
                ) for node_type in self.NODES_COLORS
            ]
            legend_arrows = [
                mpatches.FancyArrow(
                    0, 0, 1, 0,
                    facecolor=self.EDGES_COLORS[edge_type],
                    edgecolor='none',
                    label=edge_type.capitalize() if isinstance(edge_type, str) else f'Condition ({edge_type})'
                ) for edge_type in self.EDGES_COLORS
            ]

            # Draw the legend
            legend = ax.legend(
                fancybox=True,
                framealpha=0,
                fontsize='x-large',
                loc='upper right',
                handles=legend_patches + legend_arrows,
                handler_map={
                    # Patch arrows with fancy arrows in legend
                    mpatches.FancyArrow : HandlerPatch(
                        patch_func=lambda width, height, **kwargs:mpatches.FancyArrow(
                            0, 0.5*height, width, 0, width=0.2*height,
                            length_includes_head=True, head_width=height, overhang=0.5
                        )
                    ),
                }
            )
            plt.setp(legend.get_texts(), color=fontcolor)

        return ax
