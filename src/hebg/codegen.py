"""Module for code generation from HEBGraph."""

from re import sub

from typing import TYPE_CHECKING, List, Dict, Set, Tuple

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
) -> List[str]:
    """Generate codelines of the whole GeneratedBehavior from a HEBGraph.

    Args:
        graph (HEBGraph): HEBGraph to generate the behavior from.
        behaviors_histogram (Dict[Behavior, Dict[Node, int]], optional): Histogram of uses
            of all behaviors needed for the given graph. Defaults to None.
        add_dependencies (bool, optional): If True, codelines will include other GeneratedBehavior
            from sub-behaviors of the given HEBGraph and a hashmap to map behavior name
            to coresponding GeneratedBehavior. Defaults to True.

    Returns:
        List[str]: Codelines generated of the HEBGraph GeneratedBehavior.
    """
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
        class_codelines += ["from hebg.codegen import GeneratedBehavior", ""]
        for _behavior, dependency_codelines in dependencies_codelines.items():
            class_codelines += dependency_codelines

    # Class overhead
    behavior_class_name = _to_camel_case(graph.behavior.name.capitalize())
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
) -> Tuple[Dict["Behavior", List[str]], Set["Behavior"], Dict[str, str]]:
    """Parse dependencies of the given HEBGraph behavior's.

    Args:
        graph (HEBGraph): HEBGraph to parse dependecies from.
        behaviors_histogram (Dict[Behavior, Dict[Node, int]]): _description_
        add_dependencies_codelines (bool, optional): _description_. Defaults to True.

    Returns:
        Tuple of three elements:
        - dependencies_codelines (Dict[Behavior, List[str]]): Codelines of the GeneratedBehavior for
            each of the behavior used in the HEBGraph if they can be computed, else a comment.
        - behaviors_incall_codelines (Set[Behavior]): Set of behaviors that should be directly
            unrolled in the call function (because the abstraction is not worth it).
        - hashmap_codelines (List[str]): Codelines of the map between behavior names
            and coresponding GeneratedBehavior.
    """
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

        try:
            sub_graph = behavior.graph
        except NotImplementedError:
            # If subgraph cannot be computed, we simply have a ref
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
                dependencies_hashmap[behavior.name] = _to_camel_case(
                    behavior.name.capitalize()
                )
        elif n_used == 1:
            dependencies_codelines[behavior] = []
            behaviors_incall_codelines.add(behavior)

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
) -> List[str]:
    """Generate the codelines of a GeneratedBehavior call function.

    Args:
        graph (HEBGraph): HEBGraph from which to generate the behavior of.
        behaviors_incall_codelines (Set[Behavior]): Set of behavior to unroll directly
            instead of refering to.
        indent (int, optional): Indentation level. Defaults to 1.
        with_overhead (bool, optional): If True, adds the call function definition.
            Defaults to True.

    Returns:
        List[str]: Codelines of the GeneratedBehavior call function.
    """
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
) -> List[str]:
    """Generate codelines for an HEBGraph node recursively using the succesors.

    Args:
        graph (HEBGraph): HEBGraph containing the node.
        node (Node): Node to generate the call of.
        indent (int): Indentation level.
        behaviors_incall_codelines (Set[Behavior]): Set of behavior to unroll directly
            instead of refering to.
    Raises:
        NotImplementedError: Node is not an Action, FeatureCondition or Behavior.

    Returns:
        List[str]: Codelines of the node call.
    """
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


def indent_str(indent_level: int, indent_amount: int = 4) -> str:
    """Gives a string indentation from a given indent level.

    Args:
        indent_level (int): Level of indentation.
        indent_amount (int, optional): Number of spaces per indent. Defaults to 4.

    Returns:
        str: Indentation string.
    """
    return " " * indent_level * indent_amount


def _to_camel_case(text: str) -> str:
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


def _to_snake_case(text: str) -> str:
    text = text.replace("-", " ").replace("?", "")
    return "_".join(
        sub("([A-Z][a-z]+)", r" \1", sub("([A-Z]+)", r" \1", text)).split()
    ).lower()
