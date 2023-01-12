"""Module for code generation from HEBGraph."""

from re import sub
import inspect
from typing import TYPE_CHECKING, List, Dict, Set

from hebg.node import Node, Action, FeatureCondition
from hebg.behavior import Behavior
from hebg.unrolling import BEHAVIOR_SEPARATOR
from hebg.graph import get_roots, get_successors_with_index
from hebg.metrics.histograms import cumulated_hebgraph_histogram

if TYPE_CHECKING:
    from hebg.heb_graph import HEBGraph


class GeneratedBehavior:
    """Base class for generated behaviors.

    Used to reduce the overhead of abstracting a behavior."""

    def __init__(
        self,
        actions: Dict[str, "Action"] = None,
        feature_conditions: Dict[str, "FeatureCondition"] = None,
        behaviors: Dict[str, "Behavior"] = None,
    ):
        self.actions = actions if actions is not None else {}
        self.feature_conditions = feature_conditions if actions is not None else {}
        self.known_behaviors = behaviors if behaviors is not None else {}


def get_hebg_source(graph: "HEBGraph") -> str:
    """Builds the generated source code corresponding to the HEBGraph behavior.

    Args:
        graph (HEBGraph): HEBGraph to generate the source code from.

    Returns:
        str: Python source code corresponding the behavior of the HEBGraph.
    """
    class_codelines = get_behavior_class_codelines(graph)
    source = "\n".join(class_codelines)
    return source


def get_behavior_class_codelines(
    graph: "HEBGraph",
    behaviors_histogram: Dict["Behavior", Dict["Node", int]] = None,
    add_dependencies: bool = True,
):
    if behaviors_histogram is None:
        behaviors_histogram = cumulated_hebgraph_histogram(graph)

    (
        dependencies_codelines,
        behaviors_incall_codelines,
        hashmap_codelines,
    ) = get_dependencies_codelines(
        graph,
        behaviors_histogram,
        add_dependencies_codelines=add_dependencies,
    )

    class_codelines = []
    # Other behaviors dependencies
    if add_dependencies:
        for _behavior, dependency_codelines in dependencies_codelines.items():
            class_codelines += dependency_codelines

    # Class overhead
    behavior_class_name = to_camel_case(graph.behavior.name.capitalize())
    class_codelines.append(f"class {behavior_class_name}(GeneratedBehavior):")
    # Call function
    class_codelines += get_behavior_call_codelines(
        graph,
        behaviors_incall_codelines=behaviors_incall_codelines,
    )
    # Dependencies hashmap
    if add_dependencies:
        class_codelines += hashmap_codelines
    return class_codelines


def get_dependencies_codelines(
    graph: "HEBGraph",
    behaviors_histogram: Dict["Behavior", Dict["Node", int]],
    add_dependencies_codelines: bool = True,
):
    dependencies_codelines: Dict["Behavior", List[str]] = {}
    behaviors_incall_codelines: Set["Behavior"] = set()
    dependencies_hashmap: Dict[str, str] = {}
    for behavior, n_used in sorted(
        behaviors_histogram.items(), key=lambda x: x[1], reverse=True
    ):
        if not isinstance(behavior, Behavior):
            continue
        if behavior.name in graph.all_behaviors:
            behavior = graph.all_behaviors[behavior.name]
        # If subgraph cannot be computed, we simply have a ref
        try:
            sub_graph = behavior.graph
        except NotImplementedError:
            if add_dependencies_codelines:
                dependencies_codelines[behavior] = [
                    f"# Require '{behavior.name}' behavior to be given."
                ]
            continue

        if n_used > 1:
            if add_dependencies_codelines:
                dependencies_codelines[behavior] = get_behavior_class_codelines(
                    sub_graph, behaviors_histogram, add_dependencies=False
                )
                dependencies_hashmap[behavior.name] = to_camel_case(
                    behavior.name.capitalize()
                )
        elif n_used == 1:
            dependencies_codelines[behavior] = []
            behaviors_incall_codelines.add(behavior)
        else:
            raise NotImplementedError

    hashmap_codelines = []
    if dependencies_hashmap:
        hashmap_codelines = ["BEHAVIOR_TO_NAME = {"]
        hashmap_codelines += [
            f"    '{name}': {class_name},"
            for name, class_name in dependencies_hashmap.items()
        ]
        hashmap_codelines += ["}"]

    return dependencies_codelines, behaviors_incall_codelines, hashmap_codelines


def get_behavior_call_codelines(
    graph: "HEBGraph",
    behaviors_incall_codelines: Set["Behavior"],
    indent: int = 1,
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
        behaviors_incall_codelines=behaviors_incall_codelines,
    )


def get_node_call_codelines(
    graph: "HEBGraph",
    node: Node,
    indent: int,
    behaviors_incall_codelines: Set["Behavior"],
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
                    node=succ_node,
                    indent=indent + 1,
                    behaviors_incall_codelines=behaviors_incall_codelines,
                )
        return node_codelines
    if isinstance(node, Behavior):

        if node in behaviors_incall_codelines:
            if node.name in graph.all_behaviors:
                node = graph.all_behaviors[node.name]
            return get_behavior_call_codelines(
                node.graph,
                behaviors_incall_codelines=behaviors_incall_codelines,
                indent=indent,
                with_overhead=False,
            )
        default_line = f"return self.known_behaviors['{node.name}'](observation)"
        return [indent_str(indent) + default_line]
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
    raise TypeError(f"Unsupported node type: {type(node)} of {node}")


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
