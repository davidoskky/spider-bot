import pytest
from moves_exploration import (
    dof_to_move_stacked_reversibly,
)
from deck import Card

# Assuming the Card class has the necessary attributes and methods


def test_directly_movable_sequence():
    stacked_cards = [[Card(rank=2, suit=1), Card(rank=1, suit=1)]]
    top_cards = [Card(rank=3, suit=1)]
    dof_needed, _ = dof_to_move_stacked_reversibly(stacked_cards, top_cards)
    assert dof_needed == 0


def test_partially_movable_sequence():
    stacked_cards = [[Card(rank=2, suit=2)], [Card(rank=1, suit=1)]]
    top_cards = [Card(rank=3, suit=1)]
    dof_needed, dof_used = dof_to_move_stacked_reversibly(stacked_cards, top_cards)
    assert dof_needed == 1
    assert dof_used == 0


def test_non_movable_sequence():
    stacked_cards = [[Card(rank=2, suit=1)], [Card(rank=1, suit=2)]]
    top_cards = [Card(rank=4, suit=2)]
    dof_needed, dof_used = dof_to_move_stacked_reversibly(stacked_cards, top_cards)
    assert dof_needed == 2
    assert dof_used == 1


def test_empty_stacked_cards():
    stacked_cards = []
    top_cards = [Card(rank=2, suit=1)]
    dof_needed, _ = dof_to_move_stacked_reversibly(stacked_cards, top_cards)
    assert dof_needed == 0


# Add more tests as needed

# To run the tests, simply use the pytest command in the terminal.
