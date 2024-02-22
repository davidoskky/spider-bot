import logging

import pytest

from deck import Card, Deck
from moves_exploration import find_improved_equivalent_position, free_stack
from spiderSolitaire import Board, Stack
from util_tests import generate_board_from_string


def test_simple_move_one_card():
    stacks = [Stack([Card(1, 1)]) for i in range(10)]
    stacks[0] = Stack([Card(5, 1)])
    stacks[1] = Stack([Card(2, 2), Card(1, 1)])
    stacks[1].first_visible_card = 0
    stacks[2] = Stack([Card(2, 1)])
    stacks[3] = Stack([Card(4, 2)])
    stacks = tuple(stacks)
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)

    result = find_improved_equivalent_position(board)
    assert result != [], "Should not return an empty list"
    assert len(result) == 1, "Should take three moves"


def test_should_not_move():
    stacks = [Stack([Card(1, 1)]) for i in range(10)]
    stacks[0] = Stack([Card(5, 1)])
    stacks[1] = Stack([Card(2, 3), Card(1, 1)])
    stacks[1].first_visible_card = 0
    stacks[2] = Stack([Card(2, 3)])
    stacks[3] = Stack([Card(3, 2)])
    stacks = tuple(stacks)
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)

    result = find_improved_equivalent_position(board)
    assert result == [], "Should return an empty list"


def test_several_switches():
    stacks = [Stack([Card(1, 1)]) for i in range(10)]
    stacks[0] = Stack([Card(5, 1), Card(4, 1)])
    stacks[0].first_visible_card = 0
    stacks[1] = Stack([Card(1, 1), Card(6, 3), Card(5, 3), Card(4, 3), Card(3, 0)])
    stacks[1].first_visible_card = 0
    stacks[2] = Stack([Card(7, 3), Card(6, 3)])
    stacks[2].first_visible_card = 0
    stacks[3] = Stack([Card(1, 1), Card(7, 0)])
    stacks = tuple(stacks)
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)

    result = find_improved_equivalent_position(board)
    assert result != [], "Should not return an empty list"


def test_error_09_02():
    stacks = [Stack([Card(1, 1)]) for i in range(10)]
    stacks[7] = Stack(
        [Card(1, 0), Card(13, 1), Card(12, 1), Card(11, 0), Card(10, 0), Card(9, 0)]
    )
    stacks[8] = Stack([Card(8, 1)])
    stacks[9] = Stack([Card(1, 0), Card(9, 1), Card(8, 2), Card(7, 0), Card(6, 3)])

    # Set the first visible card for each stack to the index following the last hidden card (placeholder)
    for stack in stacks:
        hidden_cards = sum(
            1 for card in stack.cards if card.rank == 1 and card.suit == 0
        )
        stack.first_visible_card = hidden_cards

    board = Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)

    result = find_improved_equivalent_position(board)

    assert result == []


def test_switch2():
    stacks = [Stack([Card(1, 1)]) for i in range(10)]
    stacks[0] = Stack([Card(1, 0), Card(11, 0), Card(10, 3), Card(9, 3), Card(8, 2)])
    stacks[1] = Stack([])
    stacks[2] = Stack([Card(1, 0), Card(13, 3), Card(12, 3), Card(11, 3), Card(10, 0)])
    stacks[3] = Stack([Card(1, 0), Card(11, 2), Card(10, 2)])
    for stack in stacks:
        hidden_cards = sum(
            1 for card in stack.cards if card.rank == 1 and card.suit == 0
        )
        stack.first_visible_card = hidden_cards

    board = Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)

    result = find_improved_equivalent_position(board)
    logging.debug(f"{board.display_game_state()}")

    assert result != []


def test_valid_move_11_02():
    """
    Stack 0: XX XX XX XX 13♣ 12♦ 11♦ 10♦ 9♦ 8♦ 7♦ 6♠ 5♠ 4♠ 3♦ 2♥ 1♣ 11♣ 10♣ 9♣ 8♣ 7♣ 6♣ 5♦ 4♦ 3♦ 2♦ 1♠ 4♣ 2♠ 1♠
    Stack 1: XX XX XX 13♥ 7♠ 13♦ 12♥
    Stack 2: 13♠ 12♣ 11♠ 10♠ 9♠ 8♠
    Stack 3: 10♦ 9♦ 8♥
    Stack 4:
    Stack 5: 12♦ 11♥ 10♥ 9♥ 8♦ 7♥ 6♣ 5♥ 7♦ 6♦ 5♣ 4♣ 3♣ 2♣ 1♥ 6♥
    Stack 6: 13♣ 12♣ 11♣ 10♣ 9♣ 8♣ 7♣ 6♠ 5♠ 4♠ 3♠
    Stack 7: 5♣ 4♥
    Stack 8: 13♠ 12♠ 11♠ 10♠ 9♠ 8♠ 7♠ 6♦ 5♦ 4♦ 3♠ 2♦ 1♦
    Stack 9: 2♣ 1♣
    """
    stacks = [
        Stack([Card(1, 0), Card(1, 3), Card(4, 2), Card(2, 0), Card(1, 0)]),
        Stack([Card(1, 0), Card(13, 0), Card(7, 0), Card(13, 1), Card(12, 0)]),
        Stack(
            [Card(13, 3), Card(12, 2), Card(11, 3), Card(10, 3), Card(9, 3), Card(8, 3)]
        ),
        Stack([Card(10, 1), Card(9, 1), Card(8, 0)]),
        Stack([]),
        Stack([Card(3, 2), Card(2, 2), Card(1, 0), Card(6, 0), Card(4, 3)]),
        Stack(
            [
                Card(13, 2),
                Card(12, 2),
                Card(11, 2),
                Card(10, 2),
                Card(9, 2),
                Card(8, 2),
                Card(7, 2),
                Card(6, 0),
                Card(5, 0),
                Card(4, 0),
                Card(3, 3),
            ]
        ),
        Stack([Card(5, 2), Card(4, 0)]),
        Stack(
            [
                Card(13, 3),
                Card(12, 3),
                Card(11, 3),
                Card(10, 3),
                Card(9, 3),
                Card(8, 3),
                Card(7, 3),
                Card(6, 1),
                Card(5, 1),
                Card(4, 1),
                Card(3, 3),
                Card(2, 1),
                Card(1, 1),
            ]
        ),
        Stack([Card(2, 2), Card(1, 2)]),
    ]
    for stack in stacks:
        hidden_cards = sum(
            1 for card in stack.cards if card.rank == 1 and card.suit == 0
        )
        stack.first_visible_card = hidden_cards

    board = Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)

    result = find_improved_equivalent_position(board)

    assert result != [], "Should not produce an invalid move"


def test_error_13_02():
    """
    Stack 0: XX 12♣ 11♣ 10♥ 9♠ 8♥ 7♠
    Stack 1:
    Stack 2: XX 3♥
    Stack 3: XX 13♥ 12♥ 11♠ 10♠ 9♣ 8♦ 7♦ 6♦
    Stack 4: XX 2♠
    Stack 5: XX 3♦ 2♦
    Stack 6: XX 4♦ 3♣ 2♣
    Stack 7: XX 12♦ 11♦
    Stack 8:
    Stack 9: XX 4♠ 3♥ 2♠
    """
    stacks = [Stack([]) for _ in range(10)]  # Initialize empty stacks

    # Define the cards in each stack according to the comment
    stacks[0] = Stack(
        [
            Card(1, 0),
            Card(12, 2),
            Card(11, 2),
            Card(10, 1),
            Card(9, 3),
            Card(8, 0),
            Card(7, 3),
        ]
    )
    stacks[2] = Stack([Card(1, 0), Card(3, 1)])
    stacks[3] = Stack(
        [
            Card(1, 0),
            Card(13, 0),
            Card(12, 0),
            Card(11, 3),
            Card(10, 3),
            Card(9, 2),
            Card(8, 1),
            Card(7, 1),
            Card(6, 1),
        ]
    )
    stacks[4] = Stack([Card(1, 0), Card(2, 3)])
    stacks[5] = Stack([Card(1, 0), Card(3, 1), Card(2, 1)])
    stacks[6] = Stack([Card(1, 0), Card(4, 1), Card(3, 2), Card(2, 2)])
    stacks[7] = Stack([Card(1, 0), Card(12, 0), Card(11, 0)])
    stacks[9] = Stack([Card(1, 0), Card(4, 3), Card(3, 1), Card(2, 3)])

    for stack in stacks:
        hidden_cards = sum(
            1 for card in stack.cards if card.rank == 1 and card.suit == 0
        )
        stack.first_visible_card = hidden_cards

    board = Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)

    result = find_improved_equivalent_position(board)

    assert result != [], "A solution is available"
    # assert len(result) == 6, "The shortest solution requires 6 moves"


def test_should_move_start_of_stacked():
    """
    Stack 0: XX 10♥ 9♥ 8♥ 7♠
    Stack 1:
    Stack 2: XX 11♥ 10♥
    """
    stacks = [Stack([Card(1, 1)]) for _ in range(10)]

    # Define cards for Stack 0
    stacks[0] = Stack([Card(1, 0), Card(10, 1), Card(9, 1), Card(8, 1), Card(7, 3)])
    stacks[1] = Stack([])
    stacks[2] = Stack([Card(1, 0), Card(11, 1), Card(10, 1)])

    # Set the first visible card for stacks with hidden cards
    for stack in [stacks[0], stacks[2]]:
        stack.first_visible_card = 1  # Assuming the first card is hidden

    # Initialize the board with the defined stacks
    board = Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)

    result = find_improved_equivalent_position(board)

    assert (
        result != []
    ), "Expected to find at least one improved equivalent position but got an empty list"
    assert len(result) == 2


def test_should_move_start_of_stack():
    """
    Stack 0: 10♥ 9♥ 8♥ 7♠
    Stack 1:
    Stack 2: XX 11♥ 10♥
    """
    stacks = [Stack([Card(1, 1)]) for _ in range(10)]

    # Define cards for Stack 0
    stacks[0] = Stack([Card(10, 1), Card(9, 1), Card(8, 1), Card(7, 3)])
    stacks[1] = Stack([])
    stacks[2] = Stack([Card(1, 0), Card(11, 1), Card(10, 1)])

    stacks[0].first_visible_card = 0
    stacks[2].first_visible_card = 1

    # Initialize the board with the defined stacks
    board = Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)

    result = find_improved_equivalent_position(board)

    assert (
        result != []
    ), "Expected to find at least one improved equivalent position but got an empty list"
    assert len(result) == 2


def test_should_not_move_equivalent_start_of_stacked():
    """
    Stack 0: XX 10♥ 9♥ 8♥ 7♠
    Stack 1:
    Stack 2: XX 11♥ 10♥
    """
    stacks = [Stack([Card(1, 1)]) for _ in range(10)]

    # Define cards for Stack 0
    stacks[0] = Stack([Card(1, 0), Card(10, 1), Card(9, 1), Card(8, 1), Card(7, 3)])
    stacks[1] = Stack([])
    stacks[2] = Stack([Card(1, 0), Card(10, 1)])

    # Set the first visible card for stacks with hidden cards
    for stack in [stacks[0], stacks[2]]:
        stack.first_visible_card = 1  # Assuming the first card is hidden

    # Initialize the board with the defined stacks
    board = Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)

    result = find_improved_equivalent_position(board)

    assert result == [], "Should not move"


def test_should_not_move_equivalent_start_of_stack():
    """
    Stack 0: 10♥ 9♥ 8♥ 7♠
    Stack 1:
    Stack 2: XX 11♥ 10♥
    """
    stacks = [Stack([Card(1, 1)]) for _ in range(10)]

    # Define cards for Stack 0
    stacks[0] = Stack([Card(10, 1), Card(9, 1), Card(8, 1), Card(7, 3)])
    stacks[1] = Stack([])
    stacks[2] = Stack([Card(1, 0), Card(10, 1)])

    stacks[0].first_visible_card = 0
    stacks[2].first_visible_card = 1

    # Initialize the board with the defined stacks
    board = Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)

    result = find_improved_equivalent_position(board)

    assert result == [], "Should not move"


def test_error_14_02():
    """
    Stack 0:
    Stack 1: XX XX 1♠ 3♣ 2♣ 1♣ 9♦ 8♦ 11♠ 10♣ 9♣ 8♣ 7♣ 6♣
    Stack 2: XX XX XX XX XX 13♠ 12♠ 11♣ 10♠ 9♦ 8♦ 7♦ 6♦ 3♣ 2♣ 1♣ 4♣ 3♦ 2♦ 1♦
    Stack 3: XX 5♦ 4♣ 3♣ 2♣ 1♦
    Stack 4: XX XX 5♣ 8♠ 7♥ 6♣ 5♣ 4♣ 13♦ 12♦ 11♠ 10♠ 9♠ 8♠ 7♠ 6♥ 5♥ 4♣ 3♥ 2♣
    Stack 5:
    Stack 6:
    Stack 7: 12♣ 11♦ 10♣ 9♣ 8♣ 7♠ 6♣
    Stack 8: 13♠ 12♠ 11♠ 10♠ 9♠ 3♦ 11♦ 10♦ 9♥
    Stack 9: 6♦
    """
    stacks = [Stack([]) for _ in range(10)]

    stacks[1] = Stack(
        [Card(1, 0)]
        + [
            Card(1, 3),
            Card(3, 2),
            Card(2, 2),
            Card(1, 2),
            Card(9, 1),
            Card(8, 1),
            Card(11, 3),
            Card(10, 2),
            Card(9, 2),
            Card(8, 2),
            Card(7, 2),
            Card(6, 2),
        ]
    )
    stacks[2] = Stack(
        [Card(1, 0)]
        + [
            Card(13, 3),
            Card(12, 3),
            Card(11, 2),
            Card(10, 3),
            Card(9, 1),
            Card(8, 1),
            Card(7, 1),
            Card(6, 1),
            Card(3, 2),
            Card(2, 2),
            Card(1, 2),
            Card(4, 2),
            Card(3, 1),
            Card(2, 1),
            Card(1, 1),
        ]
    )
    stacks[3] = Stack(
        [Card(1, 0)] + [Card(5, 1), Card(4, 2), Card(3, 2), Card(2, 2), Card(1, 1)]
    )
    stacks[4] = Stack(
        [Card(1, 0)]
        + [Card(5, 2), Card(8, 3), Card(7, 0), Card(6, 2), Card(5, 2), Card(4, 2)]
        + [
            Card(13, 1),
            Card(12, 1),
            Card(11, 3),
            Card(10, 3),
            Card(9, 3),
            Card(8, 3),
            Card(7, 3),
            Card(6, 0),
            Card(5, 0),
            Card(4, 2),
            Card(3, 0),
            Card(2, 2),
        ]
    )
    stacks[7] = Stack(
        [
            Card(12, 2),
            Card(11, 1),
            Card(10, 2),
            Card(9, 2),
            Card(8, 2),
            Card(7, 3),
            Card(6, 2),
        ]
    )
    stacks[8] = Stack(
        [
            Card(13, 3),
            Card(12, 3),
            Card(11, 3),
            Card(10, 3),
            Card(9, 3),
            Card(3, 1),
            Card(11, 1),
            Card(10, 1),
            Card(9, 0),
        ]
    )
    stacks[9] = Stack([Card(6, 1)])

    for stack in stacks:
        hidden_cards = sum(
            1 for card in stack.cards if card.rank == 1 and card.suit == 0
        )
        stack.first_visible_card = hidden_cards

    # Initialize the board with the defined stacks
    board = Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)

    result = find_improved_equivalent_position(board)

    assert result != [], "Should move 7 to 4"


def test_error_22_02():
    board = generate_board_from_string(
        """Stack 0: XX XX XX XX 2♥ 1♥ 
Stack 1: XX 11♣ 4♦ 3♦ 2♦ 1♦ 
Stack 2: 6♦ 6♥ 5♥ 4♥ 3♣ 2♣ 1♣ 
Stack 3: XX XX XX XX 12♥ 11♠ 10♠ 9♠ 13♦ 12♦ 11♦ 10♥ 9♥ 8♥ 7♥ 
Stack 4: XX XX XX XX 11♥ 10♥ 
Stack 5: XX XX XX 7♦ 2♦ 10♣ 
Stack 6: XX XX XX XX 3♠ 2♠ 1♠ 4♥ 6♣ 12♦ 11♦ 10♦ 9♠ 
Stack 7: 6♦ 
Stack 8: XX XX XX 1♦ 11♥ 12♥ 11♣ 10♣ 9♣ 8♦ 7♠ 6♠ 5♠ 4♠ 3♠ 2♠ 2♣ 1♠ 
Stack 9: XX XX 13♥ 12♣ 11♠ 10♠ 
"""
    )

    result = find_improved_equivalent_position(board)
    assert result != [], "Should move 3 to 4"
    assert len(result) == 1, "Should move 3 to 4"


def test_failure_22_02():
    board = generate_board_from_string(
        """Stack 0: XX XX XX 13♠ 12♠ 11♣ 10♣ 9♣ 
Stack 1: 10♦ 4♥ 3♥ 
Stack 2: XX 13♦ 12♦ 11♦ 10♦ 9♦ 8♦ 7♦ 6♦ 5♦ 4♦ 
Stack 3: XX XX 8♣ 7♥ 3♦ 2♠ 3♠ 2♦ 1♠ 13♠ 
Stack 4: 8♠ 
Stack 5: XX XX XX 5♠ 4♦ 3♦ 2♦ 1♦ 10♠ 9♠ 8♣ 7♦ 6♣ 5♦ 1♦ 
Stack 6: XX XX XX 13♦ 12♠ 11♦ 13♣ 12♣ 11♣ 10♣ 9♥ 8♦ 7♣ 6♣ 5♣ 4♣ 3♣ 2♣ 
Stack 7: 4♠ 3♠ 2♠ 1♠ 
Stack 8: 
Stack 9: 3♣ 2♣ 1♣ 13♥ 12♥ 11♥ 10♠ 9♠ 8♠ 7♠ 6♦ """
    )

    result = find_improved_equivalent_position(board)
    assert result != [], "Should move 0 to 6"


def test_error_23():
    board = generate_board_from_string(
        """Stack 0: XX XX XX 3♥ 
Stack 1: XX XX XX XX 11♦ 
Stack 2: 
Stack 3: 13♥ 12♥ 11♣ 10♣ 9♦ 8♦ 
Stack 4: XX XX XX 7♦ 6♥ 
Stack 5: XX XX 8♠ 7♠ 6♠ 5♠ 4♠ 3♠ 2♠ 
Stack 6: 12♠ 11♠ 
Stack 7: XX XX XX XX 12♥ 11♠ 10♥ 9♥ 8♥ 7♥ 6♥ 5♣ 4♣ 3♣ 
Stack 8: XX XX XX 8♠ 7♣ 6♦ 5♦ 4♠ 3♦ 
Stack 9: """
    )

    result = find_improved_equivalent_position(board)
    assert result != [], "Should move 8 to 4"
