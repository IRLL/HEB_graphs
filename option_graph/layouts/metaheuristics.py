# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Metaheuristics used for building layouts """

import numpy as np

def simulated_annealing(initial_pos, energy, neighbor, max_iterations=1000, max_iters_without_new=100,
    iters_without_new=0, initial_temperature=5, verbose=1):

    def prob_keep(temperature, delta_e):
        return min(1, np.exp(delta_e/temperature))

    pos = initial_pos
    energy_pos = energy(pos)
    for k in range(max_iterations):
        new_pos = neighbor(pos)
        new_e = energy(new_pos)
        temperature = initial_temperature/(k+1)
        iters_without_new += 1
        prob = prob_keep(temperature, energy_pos - new_e)
        if np.random.random() < prob:
            if verbose == 1:
                print(f"{k}\t({prob:.0%})\t{energy_pos:.2f}->{new_e:.2f}", end='\r')
            pos, energy_pos = new_pos, new_e
            iters_without_new = 0

        if iters_without_new >= max_iters_without_new:
            break

    return pos
