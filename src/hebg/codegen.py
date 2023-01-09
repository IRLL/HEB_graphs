from re import sub
import inspect
from typing import TYPE_CHECKING, Optional, List, Dict, Set

from hebg.node import Node, Action, FeatureCondition
from hebg.behavior import Behavior
from hebg.unrolling import BEHAVIOR_SEPARATOR
from hebg.graph import get_roots, get_successors_with_index
from hebg.metrics.histograms import cumulated_graph_histogram

if TYPE_CHECKING:
    from hebg.heb_graph import HEBGraph


class GeneratedBehavior:
    def __init__(
        self,
        actions: Dict[str, "Action"] = None,
        feature_conditions: Dict[str, "FeatureCondition"] = None,
        behaviors: Dict[str, "Behavior"] = None,
    ):
        self.actions = actions if actions is not None else {}
        self.feature_conditions = feature_conditions if actions is not None else {}
        self.known_behaviors = behaviors if actions is not None else {}


def get_hebg_source(
    graph: "HEBGraph",
    existing_classes: Optional[Set[str]] = None,
    behaviors_histogram=None,
) -> str:
    if existing_classes is None:
        existing_classes = set()
    if behaviors_histogram is None:
        behaviors_histogram = cumulated_graph_histogram(graph)
    behavior_codelines = []
    behavior_class_name = to_camel_case(graph.behavior.name.capitalize())
    behavior_codelines.append(f"class {behavior_class_name}(GeneratedBehavior):")
    behavior_codelines += get_behavior_call_codelines(
        graph,
        behavior_codelines=behavior_codelines,
        existing_classes=existing_classes,
        behaviors_histogram=behaviors_histogram,
    )
    source = "\n".join(behavior_codelines)
    return source


def get_behavior_call_codelines(
    graph: "HEBGraph",
    behavior_codelines: List[str],
    existing_classes: Set[str],
    behaviors_histogram,
    indent: str = 1,
    with_overhead=True,
):
    call_codelines = []
    if with_overhead:
        call_codelines.append(indent_str(indent) + "def __call__(self, observation):")
        indent += 1
    roots = get_roots(graph)
    return call_codelines + get_node_call_codelines(
        graph,
        roots[0],
        indent,
        behavior_codelines=behavior_codelines,
        existing_classes=existing_classes,
        behaviors_histogram=behaviors_histogram,
    )


def get_node_call_codelines(
    graph: "HEBGraph",
    node: Node,
    indent: int,
    behavior_codelines: List[str],
    behaviors_histogram,
    existing_classes: Set[str] = None,
):
    node_codelines = []
    if isinstance(node, Action):
        action_name = node.name.split(BEHAVIOR_SEPARATOR)[-1]
        node_codelines.append(
            indent_str(indent) + f"return self.actions['{action_name}'](observation)"
        )
        return node_codelines
    if isinstance(node, FeatureCondition):
        var_name = f"edge_index_{indent-2}" if indent > 2 else "edge_index"
        fc_name = node.name.split(BEHAVIOR_SEPARATOR)[-1]
        node_codelines.append(
            indent_str(indent)
            + f"{var_name} = self.feature_conditions['{fc_name}'](observation)"
        )
        for i in [0, 1]:
            node_codelines.append(indent_str(indent) + f"if {var_name} == {i}:")
            successors = get_successors_with_index(graph, node, i)
            for succ_node in successors:
                node_codelines += get_node_call_codelines(
                    graph,
                    succ_node,
                    indent + 1,
                    behavior_codelines=behavior_codelines,
                    behaviors_histogram=behaviors_histogram,
                    existing_classes=existing_classes,
                )
        return node_codelines
    if isinstance(node, Behavior):
        sub_behavior = node
        node_codelines = [
            indent_str(indent)
            + f"return self.known_behaviors['{node.name}'](observation)"
        ]
        if node.name in graph.all_behaviors:
            sub_behavior = graph.all_behaviors[node.name]
        if node.name in existing_classes:
            return node_codelines

        try:
            sub_graph = sub_behavior.graph

            if node in behaviors_histogram and behaviors_histogram[node] == 1:
                node_codelines = get_behavior_call_codelines(
                    sub_graph,
                    behavior_codelines=behavior_codelines,
                    existing_classes=existing_classes,
                    behaviors_histogram=behaviors_histogram,
                    indent=indent,
                    with_overhead=False,
                )
                return node_codelines

            class_code = get_hebg_source(
                sub_graph, existing_classes, behaviors_histogram
            )
            behavior_codelines.insert(0, class_code)
            existing_classes.add(node.name)

        except NotImplementedError:
            behavior_codelines.insert(
                0, f"# Require '{node.name}' behavior to be given."
            )

        return node_codelines
    raise NotImplementedError


def get_instanciation(node: Node) -> str:
    if isinstance(node, Action):
        return f"{node.__class__.__name__}({node.action})"
    if isinstance(node, FeatureCondition):
        node_init_signature = inspect.signature(node.__init__)
        needed_attrs = list(node_init_signature.parameters.keys())
        attrs = {}
        for attr_name in needed_attrs:
            attr = getattr(node, attr_name)
            if isinstance(attr, str):
                attrs[attr_name] = f'"{attr}"'
            if isinstance(attr, (int, float)):
                attrs[attr_name] = attr
        attrs_str = ", ".join((f"{name}={val}" for name, val in attrs.items()))
        return f"{node.__class__.__name__}({attrs_str})"


def indent_str(indent_level: int, indent_amount: int = 4):
    return " " * indent_level * indent_amount


def to_camel_case(text: str) -> str:
    s = (
        text.replace("-", " ")
        .replace("_", " ")
        .replace("?", "")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )
    s = s.split()
    if len(text) == 0:
        return text
    return s[0] + "".join(i.capitalize() for i in s[1:])


def to_snake_case(text: str) -> str:
    text = text.replace("-", " ").replace("?", "")
    return "_".join(
        sub("([A-Z][a-z]+)", r" \1", sub("([A-Z]+)", r" \1", text)).split()
    ).lower()
