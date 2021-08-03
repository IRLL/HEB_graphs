# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Module to get pylint score. """

import sys
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

if __name__ == '__main__':
    options = [
        'option_graph',
        "--output-format=pylint_score.MyReporterClass"
    ]
    results = Run(options, exit=False)
    print(f"{results.linter.stats['global_note']:.2f}/10.00")
