# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Module containing the OptionGraph base class. """

from __future__ import annotations

from typing import Any, Dict, List
from copy import deepcopy

import networkx as nx
import numpy as np

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

from option_graph.graph import option_layout, draw_networkx_nodes_images

from option_graph.node import Node, Action, FeatureCondition, EmptyNode
from option_graph.option import Option


class OptionGraph(nx.DiGraph):

    """ Base class for option graphs.

    An OptionGraph is a DiGraph, and as such stores nodes and directed edges with
    optional data, or attributes.

    But nodes of an option graph are not arbitrary.
    Leaf nodes can either be an Action or an Option.
    Other nodes can either be a FeatureCondition or an EmptyNode.

    An OptionGraph determines the behavior of an option, it can be called with an observation
    to return the action given by this option.

    An OptionGraph edges are directed and indexed,
    this indexing for path making when calling the graph.

    As in a DiGraph loops are allowed but multiple (parallel) edges are not.

    Args:
        option: The Option object from which this graph is built.
        all_options: A dictionary of Option, this can be used to avoid cirular definitions using
            the option names as anchor instead of the Option object itself.
        any_mode: How to choose path, when multiple path are valid.
        incoming_graph_data: Additional data to include in the graph.

    """

    NODES_COLORS = {'feature_condition': 'blue', 'action': 'red',
        'option': 'orange', 'empty': None}
    EDGES_COLORS = {0:'red', 1:'green', 2:'blue', 3:'yellow',
        4:'purple', 5:'cyan', 6:'gray'}

    def __init__(self, option:Option, all_options:Dict[str, Option]=None, incoming_graph_data=None,
            any_mode:str='first', **attr):
        self.option = option
        self.all_options = all_options if all_options is not None else {}
        self.any_mode = any_mode
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)

    def add_node(self, node_for_adding:Node, **attr):
        node = node_for_adding
        super().add_node(node, type=node.type,
            color=self.NODES_COLORS[node.type], image=node.image, **attr)

    def add_edge(self, u_of_edge:Node, v_of_edge:Node, **attr):
        index = attr.pop('index', 1)
        super().add_edge(u_of_edge, v_of_edge, index=index, color=self.EDGES_COLORS[index], **attr)

    def _get_any_action(self, nodes:List[Node], observation, options_in_search:list):
        actions = []
        for node in nodes:
            action = self._get_action(node, observation, options_in_search)
            if action is None:
                return None
            actions.append(action)
        actions = [action for action in actions if action != "Impossible"]
        if len(actions) == 0:
            return "Impossible"
        if self.any_mode == 'first':
            return actions[0]
        if self.any_mode == 'last':
            return actions[-1]
        if self.any_mode == 'random':
            return np.random.choice(actions)

    def _get_action(self, node:Node, observation, options_in_search:list):
        if isinstance(node, Action):
            return node(observation)
        if isinstance(node, FeatureCondition):
            next_edge_index = node(observation)
            succs = self.successors(node)
            next_nodes = []
            for next_node in succs:
                if self.edges[node, next_node]['index'] == next_edge_index:
                    next_nodes.append(next_node)
            if len(next_nodes) == 0:
                raise IndexError(f'FeatureCondition {node} returned index {next_edge_index}'
                                 f' but {next_edge_index} was not found as an edge index')
            return self._get_any_action(next_nodes, observation, options_in_search)

        if isinstance(node, (Option, str)):
            if str(node) in options_in_search:
                return "Impossible"
            try:
                return node(observation, options_in_search)
            except NotImplementedError:
                return self.all_options[str(node)](observation, options_in_search)
        if isinstance(node, EmptyNode):
            next_node = self.successors(node).__next__()
            return self._get_action(next_node, observation, options_in_search)
        raise TypeError(f'Unknowed type {type(node)}({node}) for a node.')

    def __call__(self, observation, options_in_search=None) -> Any:
        options_in_search = [] if options_in_search is None else deepcopy(options_in_search)
        options_in_search.append(self.option.name)
        return self._get_any_action(self.roots, observation, options_in_search)

    @property
    def roots(self):
        """ Roots of the option graph (nodes without predecessors). """
        roots = []
        for node in self.nodes():
            if len(list(self.predecessors(node))) == 0:
                roots.append(node)
        return roots

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

            used_node_types = [node_type for _, node_type in self.nodes(data='type')]
            legend_patches = [
                mpatches.Patch(facecolor='none', edgecolor=color, label=node_type.capitalize()
                ) for node_type, color in self.NODES_COLORS.items()
                if node_type in used_node_types and color is not None
            ]
            used_edge_indexes = [index for _, _, index in self.edges(data='index')]
            legend_arrows = [
                mpatches.FancyArrow(0, 0, 1, 0, facecolor=color, edgecolor='none',
                    label=str(index) if index > 1 else f'{str(bool(index))} ({index})'
                ) for index, color in self.EDGES_COLORS.items()
                if index in used_edge_indexes
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
