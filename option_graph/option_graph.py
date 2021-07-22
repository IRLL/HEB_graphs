# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import networkx as nx

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

from option_graph.graph import option_layout, draw_networkx_nodes_images

class OptionGraph(nx.DiGraph):

    def add_node_feature_condition(self, node_name:str, image):
        self.add_node(node_name, type='feature_check', color='blue', image=image)

    def add_node_option(self, node_name:str, image):
        self.add_node(node_name, type='option', color='orange', image=image)

    def add_node_action(self, node_name:str, image):
        self.add_node(node_name, type='action', color='red', image=image)

    def add_node_empty(self, node_name:str):
        self.add_node(node_name, type='empty', color='purple', image=None)

    def add_edge_conditional(self, u_of_edge, v_of_edge, is_yes:bool):
        color = 'green' if is_yes else 'red'
        self.add_edge(u_of_edge, v_of_edge, type='conditional', color=color)

    def add_edge_any(self, u_of_edge, v_of_edge):
        self.add_edge(u_of_edge, v_of_edge, type='any', color='purple')

    def add_predecessors(self, prev_checks, node, force_any=False):
        if len(prev_checks) > 1 or (force_any and len(prev_checks) > 0):
            for pred in prev_checks:
                self.add_edge_any(pred, node)
        elif len(prev_checks) == 1:
            self.add_edge_conditional(prev_checks[0], node, True)

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
                mpatches.Patch(facecolor='none', edgecolor='blue', label='Feature condition'),
                mpatches.Patch(facecolor='none', edgecolor='orange', label='Option'),
                mpatches.Patch(facecolor='none', edgecolor='red', label='Action'),
            ]
            legend_arrows = [
                mpatches.FancyArrow(0, 0, 1, 0,
                    facecolor='green', edgecolor='none', label='Condition (True)'),
                mpatches.FancyArrow(0, 0, 1, 0,
                    facecolor='red', edgecolor='none', label='Condition (False)'),
                mpatches.FancyArrow(0, 0, 1, 0,
                    facecolor='purple', edgecolor='none', label='Any'),
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
