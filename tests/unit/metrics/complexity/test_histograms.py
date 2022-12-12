# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Integration tests for the hebg.metrics.complexity.histograms module. """

import pytest
import pytest_check as check

from hebg.metrics.complexity.histograms import (
    nodes_histograms,
    nodes_histogram,
    _get_node_histogram_complexity,
    _successors_by_index,
)
from hebg import Action, Behavior, FeatureCondition, HEBGraph


class TestHistograms:
    """Histograms"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize variables."""
