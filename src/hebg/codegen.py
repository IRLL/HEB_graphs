from re import sub
import inspect

from hebg.node import Node, Action, FeatureCondition
from hebg.heb_graph import HEBGraph, get_successors_with_index
from hebg.graph import get_roots


def get_hebg_source(graph: HEBGraph) -> str:
    behavior_class_codelines = []
    behavior_class_name = to_camel_case(graph.behavior.name.capitalize())
    behavior_class_codelines.append(f"class {behavior_class_name}:")

    # Init
    behavior_class_codelines.append("    def __init__(self):")
    behavior_init_codelines = [
        " " * 8 + f"self.{to_snake_case(node.name)} = " + get_instanciation(node)
        for node in graph.nodes
    ]
    behavior_class_codelines += behavior_init_codelines

    # Call
    behavior_call_codelines = ["    def __call__(self, observation):"]
    roots = get_roots(graph)

    node: Node = roots[0]
    if isinstance(node, FeatureCondition):
        for i in [0, 1]:
            behavior_call_codelines.append(
                " " * 8 + f"if self.{to_snake_case(node.name)}(observation) == {i}:"
            )
            successors = get_successors_with_index(graph, node, i)
            action: Node = successors[0]
            behavior_call_codelines.append(
                " " * 12 + f"return self.{to_snake_case(action.name)}(observation)"
            )
    if isinstance(node, Action):
        behavior_call_codelines.append(
            " " * 8 + f"return self.{to_snake_case(node.name)}(observation)"
        )

    behavior_class_codelines += behavior_call_codelines

    source = "\n".join(behavior_class_codelines)
    return source


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


def to_camel_case(text: str) -> str:
    s = text.replace("-", " ").replace("_", " ")
    s = s.split()
    if len(text) == 0:
        return text
    return s[0] + "".join(i.capitalize() for i in s[1:])


def to_snake_case(text: str) -> str:
    text = text.replace("-", " ").replace("?", " ")
    return "_".join(
        sub("([A-Z][a-z]+)", r" \1", sub("([A-Z]+)", r" \1", text)).split()
    ).lower()
