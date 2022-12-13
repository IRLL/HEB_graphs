from re import sub
import inspect

from hebg.node import Node, Action, FeatureCondition
from hebg.heb_graph import HEBGraph


def get_hebg_source(graph: HEBGraph) -> str:
    behavior_class_codelines = []
    behavior_class_name = to_camel_case(graph.behavior.name.capitalize())
    behavior_class_codelines.append(f"class {behavior_class_name}:")

    # Init
    behavior_class_codelines.append("    def __init__(self):")
    behavior_init_codelines = [
        " " * 8 + f"self.{to_snake_case(node.name)} = " + get_action_instanciation(node)
        for node in graph.nodes
    ]
    behavior_class_codelines += behavior_init_codelines

    # Call
    behavior_class_codelines.append("    def __call__(self, observation):")
    behavior_call_codelines = [
        " " * 8 + f"return self.{to_snake_case(node.name)}(observation)"
        for node in graph.nodes
    ]
    behavior_class_codelines += behavior_call_codelines

    source = "\n".join(behavior_class_codelines)
    return source


def get_action_instanciation(action: Action) -> str:
    return f"{action.__class__.__name__}({action.action})"


def to_camel_case(text: str) -> str:
    s = text.replace("-", " ").replace("_", " ")
    s = s.split()
    if len(text) == 0:
        return text
    return s[0] + "".join(i.capitalize() for i in s[1:])


def to_snake_case(text: str) -> str:
    return "_".join(
        sub(
            "([A-Z][a-z]+)", r" \1", sub("([A-Z]+)", r" \1", text.replace("-", " "))
        ).split()
    ).lower()
