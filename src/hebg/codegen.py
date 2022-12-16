from re import sub
import inspect
from typing import TYPE_CHECKING, List, Dict

from hebg.node import Node, Action, FeatureCondition
from hebg.behavior import Behavior
from hebg.graph import get_roots, get_successors_with_index

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


def get_hebg_source(graph: "HEBGraph") -> str:
    behavior_codelines = []
    behavior_class_name = to_camel_case(graph.behavior.name.capitalize())
    behavior_codelines.append(f"class {behavior_class_name}(GeneratedBehavior):")
    behavior_codelines += get_behavior_call_codelines(graph, behavior_codelines)
    source = "\n".join(behavior_codelines)
    return source


def get_behavior_call_codelines(graph: "HEBGraph", behavior_codelines: List[str]):
    indent = 1
    call_codelines = [indent_str(indent) + "def __call__(self, observation):"]
    indent += 1
    roots = get_roots(graph)
    return call_codelines + get_node_call_codelines(
        graph, roots[0], indent, behavior_codelines
    )


def get_node_call_codelines(
    graph: "HEBGraph",
    node: Node,
    indent: int,
    behavior_codelines: List[str],
):
    node_codelines = []
    if isinstance(node, Action):
        node_codelines.append(
            indent_str(indent) + f"return self.actions['{node.name}'](observation)"
        )
        return node_codelines
    if isinstance(node, FeatureCondition):
        var_name = f"edge_index_{indent-2}" if indent > 2 else "edge_index"
        node_codelines.append(
            indent_str(indent)
            + f"{var_name} = self.feature_conditions['{node.name}'](observation)"
        )
        for i in [0, 1]:
            node_codelines.append(indent_str(indent) + f"if {var_name} == {i}:")
            successors = get_successors_with_index(graph, node, i)
            for succ_node in successors:
                node_codelines += get_node_call_codelines(
                    graph, succ_node, indent + 1, behavior_codelines
                )
        return node_codelines
    if isinstance(node, Behavior):
        node_codelines.append(
            indent_str(indent)
            + f"return self.known_behaviors['{node.name}'](observation)"
        )
        behavior_codelines.insert(0, node.graph.source_code)
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
