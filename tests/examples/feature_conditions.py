from typing import Union
from enum import Enum

from hebg.node import FeatureCondition


class ThresholdFeatureCondition(FeatureCondition):
    class Relation(Enum):
        GREATER_OR_EQUAL_TO = ">="
        LESSER_OR_EQUAL_TO = "<="
        GREATER_THAN = ">"
        LESSER_THAN = "<"

    def __init__(
        self, relation: Union[Relation, str] = ">=", threshold: float = 0, **kwargs
    ) -> None:
        """Threshold-based feature condition for scalar feature."""
        self.relation = relation
        self.threshold = threshold
        self._relation = self.Relation(relation)
        display_name = self._relation.name.capitalize().replace("_", " ")
        name = f"{display_name} {threshold} ?"
        super().__init__(name=name, **kwargs)

    def __call__(self, observation: float) -> int:
        conditions = {
            self.Relation.GREATER_OR_EQUAL_TO: int(observation >= self.threshold),
            self.Relation.LESSER_OR_EQUAL_TO: int(observation <= self.threshold),
            self.Relation.GREATER_THAN: int(observation > self.threshold),
            self.Relation.LESSER_THAN: int(observation < self.threshold),
        }
        if self._relation in conditions:
            return conditions[self._relation]


class IsDivisibleFeatureCondition(FeatureCondition):
    def __init__(self, number: int = 0) -> None:
        """Is divisible feature condition for scalar feature."""
        self.number = number
        name = f"Is divisible by {number} ?"
        super().__init__(name=name, image=None)

    def __call__(self, observation: float) -> int:
        return int(observation // self.number == 1)
