import pytest
from moves_exploration import search_for_beneficial_reversible_move
from spiderSolitaire import Board, Stack
from deck import Card, Deck


def test_no_reversible_move():
    stacks = tuple([Stack([Card(1, 1)]) for i in range(10)])
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)
    assert search_for_beneficial_reversible_move(board) == False


def test_with_reversible_move():
    stacks = [Stack([Card(1, 1)]) for i in range(10)]
    stacks[1] = Stack([Card(2, 1)])
    stacks = tuple(stacks)
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)
    assert search_for_beneficial_reversible_move(board) == True


# Additional test functions for each scenario follow a similar structure
