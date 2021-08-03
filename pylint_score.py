# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Module to get pylint score. """

from pylint.lint import Run
results = Run(['option_graph'], do_exit=False)
print(f"{results.linter.stats['global_note']:.2f}/10.00")
