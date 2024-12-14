# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""Integration tests for the hebg.metrics.complexity.complexities module."""

import pytest


class TestComplexities:
    """Complexities"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize variables."""
