import logging

import pytest

from moves_exploration import (find_progressive_actions,
                               find_progressive_actions_manual)
from util_tests import generate_board_from_string


def test_algorithmic_failure_22_03():
    board = generate_board_from_string(
        """Stack 0: XX XX 3♣ 
Stack 1: XX XX XX 7♠ 8♦ 2♣ 1♣ 
Stack 2: XX XX XX 5♠ 4♥ 13♥ 12♥ 11♥ 10♥ 9♥ 8♥ 7♥ 6♥ 5♣ 4♦ 3♥ 2♥ 1♥ 
Stack 3: 8♦ 10♠ 13♥ 12♦ 11♦ 10♦ 9♦ 
Stack 4: 
Stack 5: XX 8♣ 7♣ 2♦ 1♠ 
Stack 6: XX XX XX XX 13♦ 10♦ 9♣ 8♥ 7♠ 6♠ 5♠ 4♠ 3♦ 6♦ 6♣ 5♦ 4♦ 3♦ 2♦ 1♦ 
Stack 7: XX XX 13♣ 12♣ 11♣ 10♣ 9♠ 8♣ 7♣ 2♥ 9♦ 8♠ 
Stack 8: XX XX XX 10♠ 9♠ 8♠ 1♣ 5♣ 4♣ 
Stack 9: XX XX 4♠ 3♠ 2♠ 1♠ 11♠ 13♦ 12♥ 7♦ 6♠ 13♣ 12♦
"""
    )

    result = find_progressive_actions_manual(board)

    assert result != [], "It's possible to move cards"
