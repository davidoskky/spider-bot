import logging

import pytest

from deck import Card, Deck
from moves_exploration import move_cards_removing_interfering
from spiderSolitaire import Board, Stack


def test_switch_sequences_can_sequence():
    stacks = [Stack([Card(1, 1)]) for _ in range(10)]
    stacks[0] = Stack([Card(5, 1)])
    stacks[1] = Stack([Card(3, 1), Card(2, 2), Card(1, 1)])
    stacks[1].first_visible_card = 0
    stacks[2] = Stack([Card(3, 2), Card(2, 1)])
    stacks[2].first_visible_card = 0
    stacks[3] = Stack([])
    stacks = tuple(stacks)
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)

    result = move_cards_removing_interfering(board, 1, 2, 1)
    assert result != [], "Should not return an empty list"
    assert len(result) == 3, "Should take three moves"


def test_move_card_no_clearing_needed():
    stacks = [Stack([Card(1, 1)]) for _ in range(10)]
    stacks[0] = Stack([Card(5, 1)])
    stacks[1] = Stack([Card(3, 1), Card(2, 2), Card(1, 1)])
    stacks[1].first_visible_card = 0
    stacks[2] = Stack([Card(3, 2)])
    stacks[2].first_visible_card = 0
    stacks[3] = Stack([])
    stacks = tuple(stacks)
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)

    result = move_cards_removing_interfering(board, 1, 2, 1)
    assert result != [], "Should not return an empty list"
    assert len(result) == 2, "Should take two moves"


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

    result = move_cards_removing_interfering(board, 0, 2, 2)

    assert result != []


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

    result = move_cards_removing_interfering(board, 0, 3, 4)

    logging.debug(result)
    assert result != [], "A solution is available"
    # assert len(result) == 6, "The shortest solution requires 6 moves"
