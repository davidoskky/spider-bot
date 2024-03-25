import pytest

from move_evaluator import MoveEvaluator
from spiderSolitaire import Board, InvalidMoveError, Move
from util_tests import generate_board_from_string


def test_reset():
    board = generate_board_from_string(
        """Stack 0:
    Stack 1: XX XX 8♦ 11♠ 10♣ 9♣ 8♣ 7♣ 6♣
    Stack 2: XX XX XX XX XX 1♣ 4♣ 3♦ 2♦ 1♦
    Stack 3: XX 5♦ 4♣ 3♣ 2♣ 1♦
    Stack 4: XX XX 13♦ 12♦ 11♠ 10♠ 9♠ 8♠ 7♠ 6♥ 5♥ 4♣ 3♥ 2♣
    Stack 5:
    Stack 6:
    Stack 7: 12♣ 11♦ 10♣ 9♣ 8♣ 7♠ 6♣
    Stack 8: 3♦ 11♦ 10♦ 9♥
    Stack 9: 6♦
    """
    )
    evaluator = MoveEvaluator(board)

    valid_move = Move(8, 6, 3)
    evaluator.moves_possible([valid_move])
    evaluator.reset()
    assert len(evaluator.get_moves()) == 0


def test_valid_move():
    board = generate_board_from_string(
        """Stack 0:
    Stack 1: XX XX 8♦ 11♠ 10♣ 9♣ 8♣ 7♣ 6♣
    Stack 2: XX XX XX XX XX 1♣ 4♣ 3♦ 2♦ 1♦
    Stack 3: XX 5♦ 4♣ 3♣ 2♣ 1♦
    Stack 4: XX XX 13♦ 12♦ 11♠ 10♠ 9♠ 8♠ 7♠ 6♥ 5♥ 4♣ 3♥ 2♣
    Stack 5:
    Stack 6:
    Stack 7: 12♣ 11♦ 10♣ 9♣ 8♣ 7♠ 6♣
    Stack 8: 3♦ 11♦ 10♦ 9♥
    Stack 9: 6♦
    """
    )
    evaluator = MoveEvaluator(board)
    valid_move = Move(8, 6, 3)
    assert evaluator.move_possible(valid_move), "This move is possible"
    assert evaluator.get_moves() == [valid_move], "The return value should match"


def test_invalid_move():
    board = generate_board_from_string(
        """Stack 0:
    Stack 1: XX XX 8♦ 11♠ 10♣ 9♣ 8♣ 7♣ 6♣
    Stack 2: XX XX XX XX XX 1♣ 4♣ 3♦ 2♦ 1♦
    Stack 3: XX 5♦ 4♣ 3♣ 2♣ 1♦
    Stack 4: XX XX 13♦ 12♦ 11♠ 10♠ 9♠ 8♠ 7♠ 6♥ 5♥ 4♣ 3♥ 2♣
    Stack 5:
    Stack 6:
    Stack 7: 12♣ 11♦ 10♣ 9♣ 8♣ 7♠ 6♣
    Stack 8: 3♦ 11♦ 10♦ 9♥
    Stack 9: 6♦
    """
    )
    evaluator = MoveEvaluator(board)

    invalid_move = Move(4, 1, 5)
    assert not evaluator.move_possible(invalid_move)
    assert len(evaluator.get_moves()) == 0


def test_sequence_of_moves_with_invalid_in_between():
    board = generate_board_from_string(
        """Stack 0:
    Stack 1: XX XX 8♦ 11♠ 10♣ 9♣ 8♣ 7♣ 6♣
    Stack 2: XX XX XX XX XX 1♣ 4♣ 3♦ 2♦ 1♦
    Stack 3: XX 5♦ 4♣ 3♣ 2♣ 1♦
    Stack 4: XX XX 13♦ 12♦ 11♠ 10♠ 9♠ 8♠ 7♠ 6♥ 5♥ 4♣ 3♥ 2♣
    Stack 5:
    Stack 6:
    Stack 7: 12♣ 11♦ 10♣ 9♣ 8♣ 7♠ 6♣
    Stack 8: 3♦ 11♦ 10♦ 9♥
    Stack 9: 6♦
    """
    )
    evaluator = MoveEvaluator(board)
    valid_moves = [Move(8, 6, 3), Move(8, 5, 1)]
    invalid_move = Move(4, 1, 5)
    assert evaluator.moves_possible(valid_moves)
    assert not evaluator.moves_possible([invalid_move])
    assert evaluator.get_moves() == valid_moves
