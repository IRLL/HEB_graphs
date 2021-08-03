# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Integration tests for the option_graph.metrics.complexity.histograms module. """

import pytest
import pytest_check as check

from option_graph.metrics.complexity.histograms import nodes_histograms, \
    nodes_histogram, _get_node_histogram_complexity, _successors_by_index
from option_graph import Action, Option, FeatureCondition, OptionGraph

class TestHistograms:
    """Histograms"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """ Initialize variables. """
