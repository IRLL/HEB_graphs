from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from matplotlib.axes import Axes

from networkx import (
    DiGraph,
    draw_networkx_edges,
    draw_networkx_labels,
    draw_networkx_nodes,
    ancestors,
)
import numpy as np
from hebg.behavior import Behavior
from hebg.graph import get_successors_with_index
from hebg.node import FeatureCondition, Node

if TYPE_CHECKING:
    from hebg.heb_graph import HEBGraph

Action = TypeVar("Action")


class CallGraph(DiGraph):
    def __init__(self, initial_node: "Node", heb_graph: "HEBGraph", **attr):
        super().__init__(incoming_graph_data=None, **attr)
        self.graph["n_branches"] = 0
        self.graph["n_calls"] = 0
        self.graph["frontiere"] = []
        self._known_fc: Dict[FeatureCondition, Any] = {}
        self._current_node = CallNode(0, 0)
        self.add_node(self._current_node, heb_node=initial_node, heb_graph=heb_graph)

    def call_nodes(
        self, nodes: List["Node"], observation, heb_graph: "HEBGraph"
    ) -> Action:
        self._extend_frontiere(nodes, heb_graph)
        next_call_node = self._pop_from_frontiere()
        if next_call_node is None:
            raise ValueError("No valid frontiere left in call_graph")
        return self._call_node(next_call_node, observation)

    def call_edge_labels(self):
        return [
            (self.nodes[u]["label"], self.nodes[v]["label"]) for u, v in self.edges()
        ]

    def _call_node(self, call_node: "CallNode", observation: Any) -> Action:
        node: "Node" = self.nodes[call_node]["heb_node"]
        heb_graph: "HEBGraph" = self.nodes[call_node]["heb_graph"]
        if node.type == "behavior":
            # Search for name reference in all_behaviors
            if node.name in heb_graph.all_behaviors:
                node = heb_graph.all_behaviors[node.name]
            return node(observation, self)
        elif node.type == "action":
            return node(observation)
        elif node.type == "feature_condition":
            if node in self._known_fc:
                next_edge_index = self._known_fc[node]
            else:
                next_edge_index = int(node(observation))
                self._known_fc[node] = next_edge_index
            next_nodes = get_successors_with_index(heb_graph, node, next_edge_index)
        elif node.type == "empty":
            next_nodes = list(heb_graph.successors(node))
        else:
            raise ValueError(
                f"Unknowed value {node.type} for node.type with node: {node}."
            )

        return self.call_nodes(next_nodes, observation, heb_graph=heb_graph)

    def add_node(
        self, node_for_adding, heb_node: "Node", heb_graph: "HEBGraph", **attr
    ):
        super().add_node(
            node_for_adding,
            heb_graph=heb_graph,
            heb_node=heb_node,
            label=heb_node.name,
            **attr,
        )

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        return super().add_edge(
            u_of_edge, v_of_edge, status=CallEdgeStatus.UNEXPLORED.value, **attr
        )

    def _make_new_branch(self) -> int:
        self.graph["n_branches"] += 1
        return self.graph["n_branches"]

    def _extend_frontiere(self, nodes: List["Node"], heb_graph: "HEBGraph"):
        frontiere: List[CallNode] = self.graph["frontiere"]

        parent = self._current_node
        call_nodes = []

        for i, node in enumerate(nodes):
            if i > 0:
                branch_id = self._make_new_branch()
            else:
                branch_id = parent.branch
            call_node = CallNode(branch_id, parent.rank + 1)
            self.add_node(call_node, heb_node=node, heb_graph=heb_graph)
            self.add_edge(parent, call_node)
            call_nodes.append(call_node)

        frontiere.extend(call_nodes)

    def _heb_node_from_call_node(self, node: "CallNode") -> "Node":
        return self.nodes[node]["heb_node"]

    def _pop_from_frontiere(self) -> Optional["CallNode"]:
        frontiere: List["CallNode"] = self.graph["frontiere"]

        next_node = None

        while next_node is None:
            if not frontiere:
                return None

            next_call_node = frontiere.pop(
                np.argmin(
                    [self._heb_node_from_call_node(node).cost for node in frontiere]
                )
            )
            maybe_next_node = self._heb_node_from_call_node(next_call_node)
            # Nodes should only have one parent
            parent = list(self.predecessors(next_call_node))[0]

            if isinstance(maybe_next_node, Behavior) and maybe_next_node in [
                self._heb_node_from_call_node(node)
                for node in ancestors(self, next_call_node)
            ]:
                self._update_edge_status(parent, next_call_node, CallEdgeStatus.FAILURE)
                continue

            next_node = maybe_next_node

        self.graph["n_calls"] += 1
        self.nodes[next_call_node]["call_rank"] = 1
        self._update_edge_status(parent, next_call_node, CallEdgeStatus.CALLED)
        self._current_node = next_call_node
        return next_call_node

    def _update_edge_status(
        self, start: "Node", end: "Node", status: Union["CallEdgeStatus", str]
    ):
        status = CallEdgeStatus(status)
        self.edges[start, end]["status"] = status.value

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
        draw_networkx_labels(
            self,
            labels={node: label for node, label in self.nodes(data="label")},
            ax=ax,
            pos=pos,
            **nodes_kwargs,
        )
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


class CallNode(NamedTuple):
    branch: int
    rank: int


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
    branches_per_rank: Dict[int, List[int]] = {}
    for node in call_graph.nodes():
        node: CallNode = node
        branch = node.branch
        rank = node.rank

        if rank not in branches_per_rank:
            branches_per_rank[rank] = []

        if branch not in branches_per_rank[rank]:
            branches_per_rank[rank].append(branch)

        display_branch = branches_per_rank[rank].index(branch)
        pos[node] = [display_branch, -node.rank]
    return pos
