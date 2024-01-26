# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2024 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""Metaheuristics used for building layouts"""

import numpy as np


def simulated_annealing(
    initial,
    energy,
    neighbor,
    max_iterations: int = 1000,
    initial_temperature: float = 5,
    max_iters_without_new: int = np.inf,
    verbose: bool = 1,
):
    """Perform simulated annealing metaheuristic on an energy using a neighboring.

    See https://en.wikipedia.org/wiki/Simulated_annealing for more details.

    Args:
        initial: Initial variable position.
        energy: Function giving the energy (aka cost) of a given variable position.
        neighbor: Function giving a neighbor of a given variable position.
        max_iterations: Maximum number of iterations.
        initial_temperature: Initial temperature parameter, more is more random search.
        max_iters_without_new: Maximum number of iterations without a new best position.

    """

    def prob_keep(temperature, delta_e):
        return min(1, np.exp(delta_e / temperature))

    state = initial
    energy_pos = energy(state)
    iters_without_new = 0
    for k in range(max_iterations):
        new_state = neighbor(state)
        new_energy = energy(new_state)
        temperature = initial_temperature / (k + 1)
        iters_without_new += 1
        prob = prob_keep(temperature, energy_pos - new_energy)
        if np.random.random() < prob:
            if verbose == 1:
                print(
                    f"{k}\t({prob:.0%})\t{energy_pos:.2f}->{new_energy:.2f}", end="\r"
                )
            state, energy_pos = new_state, new_energy
            iters_without_new = 0

        if iters_without_new >= max_iters_without_new:
            break
    if verbose == 1:
        print()
    return state
