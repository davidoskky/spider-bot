import logging

import pytest

from deck import Card, Deck
from moves_exploration import move_card_splitting
from spiderSolitaire import Board, Stack

# def test_require_freeing_stack():
#     stacks = [Stack([Card(1, 1)]) for i in range(10)]
#     stacks[0] = Stack([Card(5, 1)])
#     stacks[1] = Stack([Card(2, 2), Card(1, 1)])
#     stacks[1].first_visible_card = 0
#     stacks[2] = Stack([Card(3, 1)])
#     stacks[3] = Stack([Card(4, 2)])
#     stacks = tuple(stacks)
#     board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)
#
#     result = move_card_splitting(board, 1, 2, 0)
#     assert result != [], "Should not return an empty list"
#     assert len(result) == 3, "Should take three moves"


def test_direct_move():
    stacks = [Stack([Card(1, 1)]) for _ in range(10)]
    stacks[0] = Stack([Card(13, 3), Card(12, 3)])
    stacks[1] = Stack([Card(11, 3)])
    board = Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)

    result = move_card_splitting(board, 1, 0, 0)
    assert result != [], "Should not return an empty list"
    assert len(result) == 1, "Should only require one move"


def test_invalid_move():
    stacks = [Stack([Card(1, 1)]) for _ in range(10)]
    stacks[0] = Stack([Card(5, 3)])  # Source stack with a single card
    stacks[1] = Stack(
        [Card(7, 3)]
    )  # Target stack that cannot accept the card from the source stack
    board = Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)

    result = move_card_splitting(board, 0, 1, 0)  # Attempt an invalid move
    assert result == [], "Should return an empty list for an invalid move"


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
    result = move_card_splitting(board, 9, 7, 2)

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

    result = move_card_splitting(board, 0, 2, 1)
    assert result == []
