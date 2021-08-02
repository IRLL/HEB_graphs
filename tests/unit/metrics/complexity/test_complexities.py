# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Unit tests for the option_graph.metrics.complexity.general module. """

import pytest
import pytest_check as check

from option_graph.metrics.complexity.complexities import learning_complexity
from option_graph import Action, Option, FeatureCondition

class TestComplexities:
    """Complexities"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """ Initialize variables. """

        self.actions = [Action(i) for i in range(3)]
        self.feature_conditions = [FeatureCondition(str(i)) for i in range(6)]
        self.options = [Option(str(i)) for i in range(3)]
        self.used_nodes_all = {
            self.options[0]: {
                self.actions[0]: 1,
                self.actions[1]: 1,
                self.options[0]: 1,
                self.feature_conditions[0]: 1,
            },
            self.options[1]: {
                self.actions[0]: 2,
                self.actions[1]: 1,
                self.actions[2]: 1,
                self.options[0]: 1,
                self.options[1]: 1,
                self.feature_conditions[0]: 1,
                self.feature_conditions[1]: 1,
                self.feature_conditions[2]: 1,
            },
            self.options[2]: {
                self.actions[0]: 6,
                self.actions[1]: 3,
                self.actions[2]: 2,
                self.options[0]: 3,
                self.options[1]: 2,
                self.options[2]: 1,
                self.feature_conditions[0]: 3,
                self.feature_conditions[1]: 2,
                self.feature_conditions[2]: 2,
                self.feature_conditions[3]: 1,
                self.feature_conditions[4]: 1,
                self.feature_conditions[5]: 1,
            },
        }

    def test_learning_complexity_example(self):
        """ learning_complexity should give expected results on the paper handcrafted example. """
        expected_learning_complexities = {
            self.options[0]: 3,
            self.options[1]: 7,
            self.options[2]: 8,
        }
        expected_saved_complexities = {
            self.options[0]: 0,
            self.options[1]: 0,
            self.options[2]: 13,
        }
        for option in self.options:
            c_learning, saved_complexity = learning_complexity(option,
                used_nodes_all=self.used_nodes_all)

            print(option, c_learning, saved_complexity)
            check.equal(c_learning, expected_learning_complexities[option])
            check.equal(saved_complexity, expected_saved_complexities[option])
