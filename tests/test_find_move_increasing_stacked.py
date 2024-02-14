import logging

import pytest

from deck import Card, Deck
from moves_exploration import find_move_increasing_stacked_length_manual
from spiderSolitaire import Board, Stack


def test_should_move_start_of_stack():
    """
    Stack 0: 10♥ 9♥ 8♥ 7♠
    Stack 1:
    Stack 2: XX 11♥ 10♥
    """
    stacks = [Stack([Card(1, 1)]) for _ in range(10)]

    # Define cards for Stack 0
    stacks[0] = Stack([Card(10, 2), Card(9, 1), Card(8, 1), Card(7, 3)])
    stacks[1] = Stack([])
    stacks[2] = Stack([Card(1, 0), Card(11, 1), Card(10, 2)])

    stacks[0].first_visible_card = 0
    stacks[2].first_visible_card = 1

    # Initialize the board with the defined stacks
    board = Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)

    result = find_move_increasing_stacked_length_manual(board)

    assert (
        result != []
    ), "Expected to find at least one improved equivalent position but got an empty list"
    assert len(result) == 2


def test_should_move_simple():
    stacks = [Stack([Card(1, 1)]) for _ in range(10)]

    stacks[0] = Stack([Card(10, 2), Card(9, 1), Card(8, 1), Card(7, 3)])
    stacks[2] = Stack([Card(1, 0), Card(11, 1), Card(10, 2), Card(9, 1), Card(8, 2)])

    stacks[0].first_visible_card = 0
    stacks[2].first_visible_card = 1

    board = Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)

    result = find_move_increasing_stacked_length_manual(board)

    assert (
        result != []
    ), "Expected to find at least one improved equivalent position but got an empty list"
    assert len(result) == 1


def test_should_move_splitting():
    """
    Stack 0: 10♦ 9♥ 8♥ 7♥
    Stack 1: XX 10♦
    Stack 2: XX 11♥
    """
    stacks = [Stack([Card(1, 1)]) for _ in range(10)]

    stacks[0] = Stack([Card(10, 2), Card(9, 1), Card(8, 1), Card(7, 1)])
    stacks[1] = Stack([Card(1, 0), Card(10, 3)])
    stacks[2] = Stack([Card(1, 0), Card(11, 1)])

    stacks[0].first_visible_card = 0
    stacks[1].first_visible_card = 1
    stacks[2].first_visible_card = 1

    board = Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)

    result = find_move_increasing_stacked_length_manual(board)
    board.execute_moves(result)
    result2 = find_move_increasing_stacked_length_manual(board)
    result.extend(result2)

    assert (
        result != []
    ), "Expected to find at least one improved equivalent position but got an empty list"
    assert len(result) == 3


def test_should_not_move():
    """
    Stack 0: 11♥ 10♦ 9♥ 8♥ 7♥
    Stack 1: XX 10♦
    Stack 2: XX 10♦ 9♥
    """
    stacks = [Stack([Card(1, 1)]) for _ in range(10)]

    stacks[0] = Stack([Card(11, 1), Card(10, 2), Card(9, 1), Card(8, 1), Card(7, 1)])
    stacks[1] = Stack([Card(1, 0), Card(10, 3)])
    stacks[2] = Stack([Card(1, 0), Card(10, 3), Card(9, 2)])

    stacks[0].first_visible_card = 0
    stacks[1].first_visible_card = 1
    stacks[2].first_visible_card = 1

    board = Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)

    result = find_move_increasing_stacked_length_manual(board)
    board.execute_moves(result)
    result2 = find_move_increasing_stacked_length_manual(board)
    result.extend(result2)

    assert (
        result == []
    ), "Expected to find at least one improved equivalent position but got an empty list"


def test_should_not_move_from_ground():
    """
    Stack 0: 9♥ 8♥ 7♥
    Stack 1: XX 5♦
    Stack 2: XX 10♦ 9♦
    """
    stacks = [Stack([Card(1, 1)]) for _ in range(10)]

    stacks[0] = Stack([Card(9, 1), Card(8, 1), Card(7, 1)])
    stacks[1] = Stack([Card(1, 0), Card(5, 3)])
    stacks[2] = Stack([Card(1, 0), Card(10, 3), Card(9, 2)])

    stacks[0].first_visible_card = 0
    stacks[1].first_visible_card = 1
    stacks[2].first_visible_card = 1

    board = Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)

    result = find_move_increasing_stacked_length_manual(board)

    assert result != [], "No valid move"
