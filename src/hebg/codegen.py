from re import sub
import inspect
from typing import List

from hebg.node import Node, Action, FeatureCondition
from hebg.heb_graph import HEBGraph, get_successors_with_index
from hebg.graph import get_roots


def get_hebg_source(graph: HEBGraph) -> str:
    behavior_class_codelines = []
    behavior_class_name = to_camel_case(graph.behavior.name.capitalize())
    behavior_class_codelines.append(f"class {behavior_class_name}:")
    behavior_class_codelines += get_behavior_init_codelines(graph)
    behavior_class_codelines += get_behavior_call_codelines(graph)
    source = "\n".join(behavior_class_codelines)
    return source


def get_behavior_init_codelines(graph: HEBGraph) -> List[str]:
    indent = 1
    init_codelines = [indent_str(indent) + "def __init__(self):"]
    indent += 1
    init_codelines += [
        indent_str(indent)
        + f"self.{to_snake_case(node.name)} = "
        + get_instanciation(node)
        for node in graph.nodes
    ]
    return init_codelines


def get_behavior_call_codelines(graph: HEBGraph):
    indent = 1
    call_codelines = [indent_str(indent) + "def __call__(self, observation):"]
    indent += 1
    roots = get_roots(graph)

    def get_node_call_codelines(node: Node, indent: int):
        node_codelines = []
        if isinstance(node, Action):
            node_codelines.append(
                indent_str(indent)
                + f"return self.{to_snake_case(node.name)}(observation)"
            )
            return node_codelines
        if isinstance(node, FeatureCondition):
            for i in [0, 1]:
                node_codelines.append(
                    indent_str(indent)
                    + f"if self.{to_snake_case(node.name)}(observation) == {i}:"
                )
                successors = get_successors_with_index(graph, node, i)
                for succ_node in successors:
                    node_codelines += get_node_call_codelines(succ_node, indent + 1)
            return node_codelines
        raise NotImplementedError

    return call_codelines + get_node_call_codelines(roots[0], indent)


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
