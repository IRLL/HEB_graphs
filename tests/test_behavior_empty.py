# HEBGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021-2024 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""Behavior of HEBGraphs with empty nodes."""

import pytest_check as check

from hebg.node import Action
from tests.examples.behaviors import (
    E_F_A_Behavior,
    E_A_Behavior,
    F_E_A_Behavior,
    E_E_A_Behavior,
)


def test_e_a_graph():
    """(E-A) Empty nodes should skip to successor."""
    action_id = 42
    behavior = E_A_Behavior("E_A", Action(action_id))
    check.equal(behavior(None), action_id)


def test_e_f_a_graph():
    """(E-F-A) Empty should orient path properly in chain with Feature condition."""
    behavior = E_F_A_Behavior("E_F_A")
    check.equal(behavior(-1), 0)
    check.equal(behavior(1), 1)


def test_f_e_a_graph():
    """(F-E-A) Feature condition should orient path properly in chain with Empty."""
    behavior = F_E_A_Behavior("F_E_A")
    check.equal(behavior(1), 0)
    check.equal(behavior(-1), 1)


def test_e_e_a_graph():
    """(E-E-A) Empty should orient path properly in double chain."""
    behavior = E_E_A_Behavior("E_E_A")
    check.equal(behavior(None), 0)
