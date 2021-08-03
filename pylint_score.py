# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Module to get pylint score. """

import sys
from colorsys import hsv_to_rgb
import html
from pylint.interfaces import IReporter
from pylint.reporters import *
from pylint.lint import Run

class MyReporterClass(BaseReporter):
    """Report messages and layouts."""

    __implements__ = IReporter
    name = "myreporter"
    extension = "myreporter"

    def __init__(self, output=sys.stdout):
        BaseReporter.__init__(self, output)
        self.messages = []

    def handle_message(self, msg):
        """Manage message of different type and in the context of path."""
        self.messages.append(
            {
                "type": msg.category,
                "module": msg.module,
                "obj": msg.obj,
                "line": msg.line,
                "column": msg.column,
                "path": msg.path,
                "symbol": msg.symbol,
                "message": html.escape(msg.msg or "", quote=False),
                "message-id": msg.msg_id,
            }
        )

    def display_messages(self, layout):
        """Do nothing."""

    def display_reports(self, layout):
        """Do nothing."""

    def _display(self, layout):
        """Do nothing."""


def register(linter):
    """Register the reporter classes with the linter."""
    linter.register_reporter(MyReporterClass)

def interpolate(weight, x, y):
    return x * weight + (1-weight) * y

if __name__ == '__main__':
    options = [
        'option_graph',
        "--output-format=pylint_score.MyReporterClass"
    ]
    results = Run(options, exit=False)
    score = results.linter.stats['global_note']

    score_threshold = 8.0
    normalized_score = (score - score_threshold) / abs(10 - score_threshold)
    if normalized_score == 0:
        raise Exception('Insufficient score with pylint')

    if sys.argv[1] == '--score':
        print(f"{score:.2f}")
    elif sys.argv[1] == '--color':
        hsv_color = (interpolate(normalized_score, 1/3, 0), 1, 1)
        rgb_color = hsv_to_rgb(*hsv_color)
        rgb_color = tuple(int(255*value) for value in rgb_color)
        print(f"rgb{rgb_color}")
    else:
        raise ValueError(f"Unknowed argument: {sys.argv[1]}")
