from enum import Enum
from re import S
from typing import Dict, List, Optional, Tuple, Union
from matplotlib.axes import Axes

from networkx import (
    DiGraph,
    draw_networkx_edges,
    draw_networkx_labels,
    draw_networkx_nodes,
)
import numpy as np

from hebg.node import Node


class CallEdgeStatus(Enum):
    UNEXPLORED = "unexplored"
    CALLED = "called"
    FAILURE = "failure"


class CallGraph(DiGraph):
    def __init__(self, initial_node: Node, **attr):
        super().__init__(incoming_graph_data=None, **attr)
        self.graph["frontiere"] = []
        self.add_node(initial_node.name, order=0)

    def extend_frontiere(self, nodes: List[Node], parent: Node):
        frontiere: List[Node] = self.graph["frontiere"]
        frontiere.extend(nodes)

        for node in nodes:
            self.add_edge(
                parent.name, node.name, status=CallEdgeStatus.UNEXPLORED.value
            )
            node_data = self.nodes[node.name]
            parent_data = self.nodes[parent.name]
            if "order" not in node_data:
                node_data["order"] = parent_data["order"] + 1

    def pop_from_frontiere(self, parent: Node) -> Optional[Node]:
        frontiere: List[Node] = self.graph["frontiere"]

        next_node = None

        while next_node is None:
            if not frontiere:
                return None
            _next_node = frontiere.pop(np.argmin([node.cost for node in frontiere]))

            if len(list(self.successors(_next_node))) > 0:
                self.update_edge_status(parent, _next_node, CallEdgeStatus.FAILURE)
                continue

            self.update_edge_status(parent, _next_node, CallEdgeStatus.CALLED)
            next_node = _next_node

        return next_node

    def update_edge_status(
        self, start: Node, end: Node, status: Union[CallEdgeStatus, str]
    ):
        status = CallEdgeStatus(status)
        self.edges[start.name, end.name]["status"] = status.value

    def draw(
        self,
        ax: Optional[Axes] = None,
        pos: Optional[Dict[str, Tuple[float, float]]] = None,
        nodes_kwargs: Optional[dict] = None,
        label_kwargs: Optional[dict] = None,
        edges_kwargs: Optional[dict] = None,
    ):
        if pos is None:
            pos = call_graph_pos(self)
        if nodes_kwargs is None:
            nodes_kwargs = {}
        draw_networkx_nodes(self, ax=ax, pos=pos, **nodes_kwargs)
        if label_kwargs is None:
            label_kwargs = {}
        draw_networkx_labels(self, ax=ax, pos=pos, **nodes_kwargs)
        if edges_kwargs is None:
            edges_kwargs = {}
        if "connectionstyle" not in edges_kwargs:
            edges_kwargs.update(connectionstyle="arc3,rad=-0.15")
        draw_networkx_edges(
            self,
            ax=ax,
            pos=pos,
            edge_color=[
                call_status_to_color(status)
                for _, _, status in self.edges(data="status")
            ],
            **edges_kwargs,
        )


def call_status_to_color(status: Union[str, CallEdgeStatus]):
    status = CallEdgeStatus(status)
    if status is CallEdgeStatus.UNEXPLORED:
        return "black"
    if status is CallEdgeStatus.CALLED:
        return "green"
    if status is CallEdgeStatus.FAILURE:
        return "red"
    raise NotImplementedError


def call_graph_pos(call_graph: DiGraph) -> Dict[str, Tuple[float, float]]:
    pos = {}
    amount_by_order = {}
    for node, node_data in call_graph.nodes(data=True):
        order: int = node_data["order"]
        if order not in amount_by_order:
            amount_by_order[order] = 0
        else:
            amount_by_order[order] += 1
        pos[node] = [order, amount_by_order[order] / 2]
    return pos
