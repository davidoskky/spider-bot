import logging

import pytest

from moves_exploration import _move_stacked_to_temporary_position
from util_tests import generate_board_from_string


def test_clear_moving_up_and_down():
    board = generate_board_from_string(
        """Stack 0: XX 13♠ 12♠ 11♣ 10♣ 9♣ 
Stack 1: 10♦ 3♥ 
Stack 2: XX 13♦ 12♦ 11♦ 10♦ 9♦ 8♦ 7♦ 6♦ 5♦ 4♦ 
Stack 3: XX 1♠
Stack 4: 8♠ 
Stack 5: XX 1♦ 
Stack 6: XX 10♣ 9♥ 8♦ 7♣ 6♣ 5♣ 4♣ 3♣ 2♣ 
Stack 7: 4♠ 3♠ 2♠ 1♠ 
Stack 8: 
Stack 9: 1♣ 6♦ """
    )

    result = _move_stacked_to_temporary_position(board, 6, 2)
    assert result != [], "It is possible to clear this position"


def test_source_stack_is_moved():
    board = generate_board_from_string(
        """Stack 0: XX 13♠ 12♠ 11♣ 10♣ 9♣ 
Stack 1: 10♦ 3♥ 
Stack 2: XX 13♦
Stack 3: XX 1♠
Stack 4: 5♠ 
Stack 5: XX 1♦ 
Stack 6: XX 10♣ 9♥ 8♦ 7♣ 6♣ 5♣ 4♣ 3♣ 2♣ 
Stack 7: 3♠ 2♠ 1♠ 
Stack 8: 4♠ 3♠
Stack 9: 1♣ 5♦ """
    )

    result = _move_stacked_to_temporary_position(board, 8, 1)
    assert result != [], "It is possible to clear this position"


def test_error_14_02():
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

    result = _move_stacked_to_temporary_position(board, 4, 4, [7])

    assert result != [], "It is possible"
