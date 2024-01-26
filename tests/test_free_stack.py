import logging
import pytest
from moves_exploration import _move_card_to_no_intermediates, free_stack
from spiderSolitaire import Board, Stack
from deck import Card, Deck


def test_empty_board():
    stacks = tuple([Stack([]) for i in range(10)])
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)
    ignored_stacks = []
    result = free_stack(board, ignored_stacks)
    assert result == []


def test_several_simple_options():
    stacks = [Stack([]) for i in range(10)]
    stacks[0] = Stack([Card(1, 1)])
    stacks[1] = Stack([Card(2, 2)])
    stacks[2] = Stack([Card(3, 1)])
    stacks[3] = Stack([Card(4, 2)])
    stacks = tuple(stacks)
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)

    ignored_stacks = []
    result = free_stack(board, ignored_stacks)
    assert result != [], "Should not return an empty list"
    assert len(result) == 1, "Should return one stack ID"


def test_one_stack_can_be_freed_easily():
    stacks = [Stack([]) for i in range(10)]
    stacks[0] = Stack([Card(1, 1)])
    stacks[1] = Stack([Card(2, 2)])
    stacks = tuple(stacks)
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)

    ignored_stacks = []
    result = free_stack(board, ignored_stacks)
    assert result != [], "Should not return an empty list"
    assert len(result) == 1, "Should return one stack ID"


def test_one_stack_can_be_freed_easily_reordered():
    stacks = [Stack([]) for i in range(10)]
    stacks[0] = Stack([])
    stacks[1] = Stack([Card(2, 2)])
    stacks[2] = Stack([Card(1, 1)])
    stacks = tuple(stacks)
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)

    ignored_stacks = []
    result = free_stack(board, ignored_stacks)
    assert result != [], "Should not return an empty list"
    assert len(result) == 1, "Should return one stack ID"


def test_one_stack_can_be_freed_many_moves():
    stacks = [Stack([Card(1, 1)]) for i in range(10)]
    stacks[0] = Stack([Card(4, 1), Card(3, 2), Card(2, 1), Card(1, 2)])
    stacks[0].first_visible_card = 0
    stacks[1] = Stack([Card(5, 2)])
    stacks[2] = Stack([])
    stacks[3] = Stack([])
    stacks = tuple(stacks)
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)

    ignored_stacks = []
    result = free_stack(board, ignored_stacks)

    for move in result:
        board.move_by_index(*move)
    assert result != [], "Should not return an empty list"
    assert board.count_empty_stacks() == 3


def test_one_stack_can_be_freed_intermediate_change():
    stacks = [Stack([Card(1, 1)]) for i in range(10)]
    stacks[0] = Stack(
        [Card(6, 1), Card(5, 2), Card(4, 1), Card(3, 2), Card(2, 1), Card(1, 2)]
    )
    stacks[0].first_visible_card = 0
    stacks[1] = Stack([Card(5, 2)])
    stacks[2] = Stack([])
    stacks[3] = Stack([])
    stacks[4] = Stack([Card(7, 2)])
    stacks = tuple(stacks)
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)

    ignored_stacks = []
    result = free_stack(board, ignored_stacks)

    for move in result:
        board.move_by_index(*move)
    assert result != [], "Should not return an empty list"
    assert board.count_empty_stacks() == 3


def test_complex_splitting():
    stacks = [Stack([Card(1, 1)]) for i in range(10)]
    stacks[0] = Stack(
        [Card(6, 1), Card(5, 2), Card(4, 2), Card(3, 3), Card(2, 1), Card(1, 2)]
    )
    stacks[0].first_visible_card = 0
    stacks[1] = Stack([Card(5, 2)])
    stacks[2] = Stack([])
    stacks[3] = Stack([])
    stacks[4] = Stack([Card(7, 2)])
    stacks = tuple(stacks)
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)

    ignored_stacks = []
    result = free_stack(board, ignored_stacks)

    for move in result:
        board.move_by_index(*move)
    assert result != [], "Should not return an empty list"
    assert board.count_empty_stacks() == 3


def test_no_freeable_stack():
    stacks = tuple([Stack([Card(1, 1)]) for i in range(10)])
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)
    ignored_stacks = []
    result = free_stack(board, ignored_stacks)
    assert result == []


def test_identified_1():
    stacks = [Stack([Card(1, 1)]) for i in range(10)]
    stacks[0] = Stack([Card(5, 2), Card(4, 1), Card(3, 2)])
    stacks[0].first_visible_card = 0
    stacks[1] = Stack([])
    stacks[2] = Stack([Card(2, 1), Card(1, 2)])
    stacks[2].first_visible_card = 0
    stacks[3] = Stack([])
    stacks = tuple(stacks)
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)

    ignored_stacks = []
    result = free_stack(board, ignored_stacks)

    for move in result:
        board.move_by_index(*move)
    assert result != [], "Should not return an empty list"
    assert board.count_empty_stacks() == 3


def test_identified_1_intermediate():
    stacks = [Stack([Card(1, 1)]) for i in range(10)]
    stacks[0] = Stack([Card(5, 2), Card(4, 1), Card(3, 2)])
    stacks[0].first_visible_card = 0
    stacks[1] = Stack([])
    stacks[2] = Stack([Card(2, 1), Card(1, 2)])
    stacks[2].first_visible_card = 0
    stacks[3] = Stack([])
    stacks = tuple(stacks)
    board = Board(stacks=stacks, deck=Deck(), completed_stacks=0)

    result = _move_card_to_no_intermediates(board, 2, 0, 0)
    logging.debug(f"{result}")

    for move in result:
        board.move_by_index(*move)
    assert result != [], "Should not return an empty list"
    assert board.count_empty_stacks() == 2
