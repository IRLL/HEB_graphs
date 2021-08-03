# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Module to get coverage score. """

import sys
from xml.dom import minidom
from utils import score_to_rgb_color

if __name__ == '__main__':
    file = minidom.parse('coverage.xml')
    coverage = file.getElementsByTagName('coverage')
    coverage = float(coverage[0].attributes['line-rate'].value)
    color = score_to_rgb_color(coverage, 0, 1)
    if sys.argv[1] == '--score':
        print(f"{coverage:.1%}")
    elif sys.argv[1] == '--color':
        print(color)
    else:
        raise ValueError(f"Unknowed argument: {sys.argv[1]}")
