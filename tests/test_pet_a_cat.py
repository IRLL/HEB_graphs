"""This examples shows how could we hierarchicaly build a behavior to pet a cat.

Here is the hierarchical structure that we would want:

```
PetACat:
    IsThereACatAround ?
    -> Yes:
        PetNearbyCat
    -> No:
        LookForACat

PetNearbyCat:
    IsYourHandNearTheCat ?
    -> Yes:
        Pet
    -> No:
        MoveYourHandNearTheCat
```

"""

import pytest
import pytest_check as check

import matplotlib.pyplot as plt

from hebg import HEBGraph, Action, FeatureCondition, Behavior
from hebg.unrolling import unroll_graph
from tests.test_code_generation import _unidiff_output


class Pet(Action):
    def __init__(self) -> None:
        super().__init__(action="Pet")


class IsYourHandNearTheCat(FeatureCondition):
    def __init__(self, hand) -> None:
        super().__init__(name="Is hand near the cat ?")
        self.hand = hand

    def __call__(self, observation):
        # Could be a very complex function that returns 1 is the hand is near the cat else 0.
        if observation["cat"] == observation[self.hand]:
            return int(True)  # 1
        return int(False)  # 0


class MoveYourHandNearTheCat(Behavior):
    def __init__(self) -> None:
        super().__init__(name="Move slowly your hand near the cat")

    def __call__(self, observation, *args, **kwargs) -> Action:
        # Could be a very complex function that returns actions from any given observation
        return Action("Move hand to cat")


class PetNearbyCat(Behavior):
    def __init__(self) -> None:
        super().__init__(name="Pet nearby cat")

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        is_hand_near_cat = IsYourHandNearTheCat(hand="hand")
        graph.add_edge(is_hand_near_cat, MoveYourHandNearTheCat(), index=int(False))
        graph.add_edge(is_hand_near_cat, Pet(), index=int(True))

        return graph


class IsThereACatAround(FeatureCondition):
    def __init__(self) -> None:
        super().__init__(name="Is there a cat around ?")

    def __call__(self, observation):
        # Could be a very complex function that returns 1 is there is a cat around else 0.
        if "cat" in observation:
            return int(True)  # 1
        return int(False)  # 0


class LookForACat(Behavior):
    def __init__(self) -> None:
        super().__init__(name="Look for a nearby cat")

    def __call__(self, observation, *args, **kwargs) -> Action:
        # Could be a very complex function that returns actions from any given observation
        return Action("Move to a cat")


class PetACat(Behavior):
    def __init__(self) -> None:
        super().__init__(name="Pet a cat")

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        is_a_cat_around = IsThereACatAround()
        graph.add_edge(is_a_cat_around, LookForACat(), index=int(False))
        graph.add_edge(is_a_cat_around, PetNearbyCat(), index=int(True))
        return graph


class TestPetACat:
    """PetACat example"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.pet_nearby_cat_behavior = PetNearbyCat()
        self.pet_a_cat_behavior = PetACat()

    def test_call(self):
        """should give expected call"""
        observation = {
            "cat": "sofa",
            "hand": "computer",
        }

        # Call on observation
        action = self.pet_a_cat_behavior(observation)
        check.equal(action, Action("Move hand to cat"))

    def test_pet_a_cat_graph_edges(self):
        """should give expected edges for PetACat"""
        # Obtain networkx graph
        graph = self.pet_a_cat_behavior.graph
        check.equal(
            set(graph.edges(data="index")),
            {
                ("Is there a cat around ?", "Look for a nearby cat", 0),
                ("Is there a cat around ?", "Pet nearby cat", 1),
            },
        )

    def test_pet_nearby_cat_graph_edges(self):
        """should give expected edges for PetNearbyCat"""
        # Obtain networkx graph
        graph = self.pet_nearby_cat_behavior.graph
        check.equal(
            set(graph.edges(data="index")),
            {
                ("Is hand near the cat ?", "Move slowly your hand near the cat", 0),
                ("Is hand near the cat ?", "Action(Pet)", 1),
            },
        )

    def test_draw(self):
        """should be able to draw without error"""
        fig, ax = plt.subplots()
        self.pet_a_cat_behavior.graph.draw(ax)
        plt.close(fig)

    def test_draw_unrolled(self):
        """should be able to draw without error"""
        fig, ax = plt.subplots()
        unrolled_graph = unroll_graph(self.pet_a_cat_behavior.graph)
        unrolled_graph.draw(ax)
        plt.close(fig)

    @pytest.mark.filterwarnings("ignore:Could not load graph for behavior")
    def test_codegen(self):
        """should generate expected source code"""
        code = self.pet_a_cat_behavior.graph.generate_source_code()
        expected_code = "\n".join(
            (
                "from hebg.codegen import GeneratedBehavior",
                "",
                "# Require 'Look for a nearby cat' behavior to be given.",
                "# Require 'Move slowly your hand near the cat' behavior to be given.",
                "class PetACat(GeneratedBehavior):",
                "    def __call__(self, observation):",
                "        edge_index = self.feature_conditions['Is there a cat around ?'](observation)",
                "        if edge_index == 0:",
                "            return self.known_behaviors['Look for a nearby cat'](observation)",
                "        if edge_index == 1:",
                "            edge_index_1 = self.feature_conditions['Is hand near the cat ?'](observation)",
                "            if edge_index_1 == 0:",
                "                return self.known_behaviors['Move slowly your hand near the cat'](observation)",
                "            if edge_index_1 == 1:",
                "                return self.actions['Action(Pet)'](observation)",
            )
        )
        check.equal(code, expected_code, _unidiff_output(code, expected_code))
