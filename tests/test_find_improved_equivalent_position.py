import logging

import pytest

from deck import Card, Deck
from moves_exploration import find_improved_equivalent_position_manual
from spiderSolitaire import Board, Stack


def test_simple_move_one_card():
    stacks = [Stack([Card(1, 1)]) for i in range(10)]
    stacks[0] = Stack([Card(5, 1)])
    stacks[1] = Stack([Card(2, 2), Card(1, 1)])
    stacks[1].first_visible_card = 0
    stacks[2] = Stack([Card(2, 1)])
    stacks[3] = Stack([Card(4, 2)])
    stacks = tuple(stacks)
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)

    result = find_improved_equivalent_position_manual(board)
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

    result = find_improved_equivalent_position_manual(board)
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

    result = find_improved_equivalent_position_manual(board)
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

    result = find_improved_equivalent_position_manual(board)

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

    result = find_improved_equivalent_position_manual(board)
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

    result = find_improved_equivalent_position_manual(board)

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

    result = find_improved_equivalent_position_manual(board)

    assert result != [], "A solution is available"
    # assert len(result) == 6, "The shortest solution requires 6 moves"
