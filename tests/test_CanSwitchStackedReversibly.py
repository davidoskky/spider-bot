from moves_exploration import can_move_stacked_reversibly, can_switch_stacked_reversibly
from deck import Card
import pytest


def test_empty_stacks():
    # Test with empty stacks
    assert not can_switch_stacked_reversibly([], [], [], 0)


def test_invalid_empty_stacks_argument():
    # Test with invalid empty_stacks argument
    with pytest.raises(ValueError):
        can_switch_stacked_reversibly([], [], [], -1)


def test_single_sequence_switch():
    # Test switching a single non-empty sequence with an empty sequence
    card1 = Card(1, 1)
    card2 = Card(2, 1)
    assert can_switch_stacked_reversibly([[card1]], [], [], 1)


def test_single_sequence_switch_no_dof():
    # Test switching a single non-empty sequence with an empty sequence
    card1 = Card(1, 1)
    assert can_switch_stacked_reversibly([[card1]], [], [], 0)


def test_impossible_switch_due_to_insufficient_dof():
    # Test a scenario where switch is not possible due to insufficient degrees of freedom
    card1 = Card(2, 1)
    card2 = Card(2, 2)
    assert not can_switch_stacked_reversibly([[card1]], [[card2]], [], 0)


def test_possible_switch_with_sufficient_dof():
    # Test a scenario where switch is possible with sufficient degrees of freedom
    card1 = Card(2, 2)
    card2 = Card(2, 1)
    assert can_switch_stacked_reversibly([[card1]], [[card2]], [], 1)


def test_possible_switch_with_sufficient_dof_multiple_cards():
    # Test a scenario where switch is possible with sufficient degrees of freedom
    cards1 = [[Card(3, 1)], [Card(2, 2)]]
    card2 = Card(3, 3)
    assert can_switch_stacked_reversibly(cards1, [[card2]], [Card(4, 2)], 1)
