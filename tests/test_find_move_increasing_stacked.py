import logging

import pytest

from deck import Card, Deck
from moves_exploration import (find_improved_equivalent_position,
                               find_move_increasing_stacked_length_manual,
                               free_stack)
from spiderSolitaire import Board, Stack
from util_tests import generate_board_from_string


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
    stacks[2] = Stack([Card(1, 0), Card(10, 3), Card(9, 3)])

    stacks[0].first_visible_card = 0
    stacks[1].first_visible_card = 1
    stacks[2].first_visible_card = 1

    board = Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)

    result = find_move_increasing_stacked_length_manual(board)

    assert result == [], "There is no valid move"


def test_raise_error_15_02():
    board = generate_board_from_string(
        """Stack 0: XX XX XX XX XX 6♥ 5♣ 5♦ 4♦ 3♦ 12♥ 11♠ 3♥ 
Stack 1: XX 13♦ 12♦ 11♦ 12♦ 
Stack 2: XX XX 7♦ 6♥ 5♥ 4♥ 3♦ 
Stack 3: XX XX XX XX 1♣ 9♥ 8♣ 2♠ 1♠ 
Stack 4: XX 8♥ 13♣ 12♠ 
Stack 5: XX XX XX 13♣ 12♣ 11♣ 10♦ 9♠ 8♠ 
Stack 6: 6♦ 5♣ 4♣ 3♣ 2♣ 1♣ 
Stack 7: XX XX XX XX 2♥ 1♦ 1♥ 7♥ 
Stack 8: XX XX XX 13♥ 12♥ 11♦ 10♠ 9♠ 8♠ 9♦ 8♦ 
Stack 9: XX XX XX XX 5♠ 3♠ 2♠ 1♦ 2♦ 8♣ 7♠ 6♠ 
"""
    )
    result = find_move_increasing_stacked_length_manual(board)

    assert result != []


def test_should_not_move_sequence():
    board = generate_board_from_string(
        """Stack 0: XX XX XX 1♠ 1♥ 
Stack 1: XX XX XX XX XX 5♣ 8♦ 7♣ 6♥ 1♦ 8♥ 
Stack 2: XX XX XX XX XX 13♥ 12♣ 11♣ 10♦ 13♥ 12♦ 4♦ 7♠ 6♥ 5♠ 
Stack 3: XX XX XX XX XX 2♥ 1♦ 13♠ 12♠ 
Stack 4: XX XX XX XX 7♦ 13♣ 12♥ 
Stack 5: XX 10♦ 4♥ 3♥ 2♠ 1♣ 10♣ 9♥ 8♦ 
Stack 6: XX 3♦ 
Stack 7: XX XX 7♣ 13♦ 10♣ 
Stack 8: XX XX XX XX 5♥ 13♣ 6♦ 5♣ 
Stack 9: XX XX XX XX 10♥ 9♥ 8♠ 7♠ 6♠ 12♣ 11♦ 10♠ 9♠ 
"""
    )

    result = find_move_increasing_stacked_length_manual(board)

    assert result == [], "The sequence should not be split"


def test_should_not_move_sequence_2():
    board = generate_board_from_string(
        """Stack 0: XX XX 6♥ 
Stack 1: XX XX XX XX XX 12♠ 11♠ 10♠ 9♠ 
Stack 2: XX XX XX XX XX 13♦ 12♦ 11♥ 10♦ 9♦ 8♦ 7♣ 
Stack 3: XX XX XX XX XX 4♣ 3♣ 2♣ 
Stack 4: XX XX XX 13♥ 12♦ 11♣ 10♣ 9♣ 8♣ 7♣ 
Stack 5: XX 6♣ 13♥ 12♥ 11♥ 10♠ 9♠ 
Stack 6: 
Stack 7: 4♣ 
Stack 8: XX XX XX XX 2♠ 1♠ 9♥ 8♥ 7♥ 6♥ 5♥ 4♠ 
Stack 9: 4♦ 3♦ 
"""
    )

    result = find_move_increasing_stacked_length_manual(board)

    assert result == [], "The sequence should not be split"
