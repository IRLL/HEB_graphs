# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

from __future__ import annotations

from typing import Any, Dict

import networkx as nx
from copy import deepcopy

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

from option_graph.graph import option_layout, draw_networkx_nodes_images

from option_graph.node import Node, Action, FeatureCondition, EmptyNode
from option_graph.option import Option


class OptionGraph(nx.DiGraph):

    NODES_COLORS = {'feature_condition': 'blue', 'action': 'red',
        'option': 'orange', 'empty': 'purple'}
    EDGES_COLORS = {0:'red', 1:'green', 2:'blue', 3:'yellow',
        4:'rose', 5:'cyan', 6:'gray', 'any':'purple'}
    ANY_MODES = ['first', 'random']

    def __init__(self, option:Option, all_options:Dict[str, Option], incoming_graph_data,
            any_mode:str='first', **attr):
        self.option = option
        self.all_options = all_options
        self.any_mode = any_mode
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)

    def add_node(self, node:Node):
        super().add_node(node.name, type=node.type, object=node,
            color=self.NODES_COLORS[node.type], image=node.image)

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
            self.add_edge_condition(prev_checks[0], node, int(True))

    def _next(self, node:Node, observation, options_in_search:list):
        if isinstance(node, Action):
            return node(observation)
        if isinstance(node, FeatureCondition):
            next_edge_index = node(observation)
            succs = self.successors(node.name)
            for succ in succs:
                if self.edges[node.name, succ]['index'] == next_edge_index:
                    next_node = self.nodes[node.name]['object']
                    return self._next(next_node, observation, options_in_search)
            raise IndexError(f'FeatureCondition {node} returned index {next_edge_index}'
                             f' but {next_edge_index} was not found as an edge index')
        if isinstance(node, Option):
            try:
                return self.nodes[node.name]['object'](observation, options_in_search)
            except NotImplementedError:
                return self.all_options[node.name](observation, options_in_search)

    def __call__(self, observation, options_in_search=None) -> Any:
        if options_in_search is None:
            options_in_search = []
        else:
            options_in_search = deepcopy(options_in_search)
        options_in_search.append(self.option.name)

        raise NotImplementedError

    def draw(self, ax:Axes, **kwargs) -> Axes:
        """ Draw the OptionGraph on the given Axis.

        Args:
            ax: The matplotlib ax to draw on.

        Kwargs:
            fontcolor: Font color to use for all texts.

        Returns:
            The resulting matplotlib Axis drawn on.

        """
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
                mpatches.Patch(facecolor='none', edgecolor=color, label=node_type.capitalize()
                ) for node_type, color in self.NODES_COLORS.items()
            ]
            legend_arrows = [
                mpatches.FancyArrow(0, 0, 1, 0, facecolor=color, edgecolor='none',
                    label=edge_type.capitalize() if isinstance(edge_type, str) \
                        else f'Condition ({edge_type})'
                ) for edge_type, color in self.EDGES_COLORS.items()
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
