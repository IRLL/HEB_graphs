# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Unit tests for the option_graph.layouts.metaheuristics module. """

import numpy as np

import pytest_check as check

from option_graph.layouts.metaheuristics import simulated_annealing


def test_simulated_annealing():
    """Simulated annealing must work on the simple x**2 case."""

    def energy(x):
        return x ** 2

    step_size = 0.05

    def neighbor(x):
        return x + np.random.choice([-1, 1]) * step_size

    optimal_x = simulated_annealing(
        -1, energy, neighbor, max_iterations=1000, initial_temperature=5, verbose=1
    )

    check.less_equal(abs(optimal_x), 3.1 * step_size)
