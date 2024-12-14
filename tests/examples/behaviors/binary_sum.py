from hebg import Action
from tests.examples.behaviors import F_A_Behavior
from tests.examples.feature_conditions import IsDivisibleFeatureCondition


def build_binary_sum_behavior() -> F_A_Behavior:
    feature_condition = IsDivisibleFeatureCondition(2)
    actions = {0: Action(0), 1: Action(1)}
    binary_1 = F_A_Behavior("Is x1 in binary ?", feature_condition, actions)

    feature_condition = IsDivisibleFeatureCondition(2)
    actions = {0: Action(1), 1: Action(0)}
    binary_0 = F_A_Behavior("Is x0 in binary ?", feature_condition, actions)

    feature_condition = IsDivisibleFeatureCondition(4)
    actions = {0: Action(0), 1: binary_1}
    binary_11 = F_A_Behavior("Is x11 in binary ?", feature_condition, actions)

    feature_condition = IsDivisibleFeatureCondition(4)
    actions = {0: binary_0, 1: binary_1}
    binary_10_01 = F_A_Behavior("Is x01 or x10 in binary ?", feature_condition, actions)

    feature_condition = IsDivisibleFeatureCondition(8)
    actions = {0: binary_11, 1: binary_10_01}

    return F_A_Behavior("Is sum (of last 3 binary) 2 ?", feature_condition, actions)
