from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypeVar, Union
from matplotlib.axes import Axes

from networkx import (
    DiGraph,
    draw_networkx_edges,
    draw_networkx_labels,
    draw_networkx_nodes,
)
import numpy as np
from hebg.behavior import Behavior
from hebg.graph import get_successors_with_index
from hebg.node import FeatureCondition, Node

if TYPE_CHECKING:
    from hebg.heb_graph import HEBGraph

Action = TypeVar("Action")


class CallGraph(DiGraph):
    def __init__(self, initial_node: "Node", **attr):
        super().__init__(incoming_graph_data=None, **attr)
        self.graph["n_calls"] = 0
        self.graph["frontiere"] = []
        self._known_fc: Dict[FeatureCondition, Any] = {}
        self.add_node(initial_node.name, exploration_order=0, calls_order=[0])

    def call_nodes(
        self,
        nodes: List["Node"],
        observation,
        hebgraph: "HEBGraph",
        parent: "Node" = None,
    ) -> Action:
        self._extend_frontiere(nodes, parent)
        next_node = self._pop_from_frontiere(parent)
        if next_node is None:
            raise ValueError("No valid frontiere left in call_graph")
        return self._call_node(next_node, observation, hebgraph)

    def _call_node(
        self,
        node: "Node",
        observation: Any,
        hebgraph: "HEBGraph",
    ) -> Action:
        if node.type == "behavior":
            # Search for name reference in all_behaviors
            if node.name in hebgraph.all_behaviors:
                node = hebgraph.all_behaviors[node.name]
            return node(observation, self)
        elif node.type == "action":
            return node(observation)
        elif node.type == "feature_condition":
            if node in self._known_fc:
                next_edge_index = self._known_fc[node]
            else:
                next_edge_index = int(node(observation))
                self._known_fc[node] = next_edge_index
            next_nodes = get_successors_with_index(hebgraph, node, next_edge_index)
        elif node.type == "empty":
            next_nodes = list(hebgraph.successors(node))
        else:
            raise ValueError(
                f"Unknowed value {node.type} for node.type with node: {node}."
            )

        return self.call_nodes(
            next_nodes,
            observation,
            hebgraph=hebgraph,
            parent=node,
        )

    def _extend_frontiere(self, nodes: List["Node"], parent: "Node"):
        frontiere: List["Node"] = self.graph["frontiere"]
        frontiere.extend(nodes)

        for node in nodes:
            self.add_edge(
                parent.name, node.name, status=CallEdgeStatus.UNEXPLORED.value
            )
            node_data = self.nodes[node.name]
            parent_data = self.nodes[parent.name]
            if "exploration_order" not in node_data:
                node_data["exploration_order"] = parent_data["exploration_order"] + 1

    def _pop_from_frontiere(self, parent: "Node") -> Optional["Node"]:
        frontiere: List["Node"] = self.graph["frontiere"]

        next_node = None

        while next_node is None:
            if not frontiere:
                return None
            _next_node = frontiere.pop(np.argmin([node.cost for node in frontiere]))

            if (
                isinstance(_next_node, Behavior)
                and len(list(self.successors(_next_node))) > 0
            ):
                self._update_edge_status(parent, _next_node, CallEdgeStatus.FAILURE)
                continue

            next_node = _next_node

        self.graph["n_calls"] += 1
        calls_order = self.nodes[next_node.name].get("calls_order", None)
        if calls_order is None:
            calls_order = []
        calls_order.append(self.graph["n_calls"])
        self.nodes[next_node.name]["calls_order"] = calls_order
        self._update_edge_status(parent, next_node, CallEdgeStatus.CALLED)
        return next_node

    def _update_edge_status(
        self, start: "Node", end: "Node", status: Union["CallEdgeStatus", str]
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
            pos = _call_graph_pos(self)
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
                _call_status_to_color(status)
                for _, _, status in self.edges(data="status")
            ],
            **edges_kwargs,
        )


class CallEdgeStatus(Enum):
    UNEXPLORED = "unexplored"
    CALLED = "called"
    FAILURE = "failure"


def _call_status_to_color(status: Union[str, "CallEdgeStatus"]):
    status = CallEdgeStatus(status)
    if status is CallEdgeStatus.UNEXPLORED:
        return "black"
    if status is CallEdgeStatus.CALLED:
        return "green"
    if status is CallEdgeStatus.FAILURE:
        return "red"
    raise NotImplementedError


def _call_graph_pos(call_graph: DiGraph) -> Dict[str, Tuple[float, float]]:
    pos = {}
    amount_by_order = {}
    for node, node_data in call_graph.nodes(data=True):
        order: int = node_data["exploration_order"]
        if order not in amount_by_order:
            amount_by_order[order] = 0
        else:
            amount_by_order[order] += 1
        pos[node] = [order, amount_by_order[order] / 2]
    return pos
