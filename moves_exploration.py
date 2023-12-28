from __future__ import annotations
from collections import deque
from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from spiderSolitaire import Board


DEFAULT_WEIGHTS = {
    "visible_card_weight": 129.31059791,
    "hidden_card_weight": 87.0175388,
    "breaking_stackable_weight": -369.45669462,
    "breaking_sequence_weight": 138.15459622,
    "empty_stack_weight": 3000.2845636,
    "count_completed_stacks": 10000.70034362,
    "sequence_length_weight": 359.84705868,
    "stacked_length_weight": 65.9502677,
    "total_rank_sequence_weight": 179.44314697,
    "total_rank_stacked_weight": -327.68215433,
    "stacked_length_indicator": 179.52688487,
    "sequence_length_indicator": 340.70034362,
}


def score_board(board, weights) -> int:
    """
    Evaluate and score the board state based on various criteria.

    :param board: The current state of the Spider Solitaire game.
    :param weights: A dictionary of weights for different scoring criteria.
    :return: The score of the board.
    """
    score: int = 0

    score += weights["visible_card_weight"] * board.count_visible_cards()

    score -= weights["hidden_card_weight"] * board.count_hidden_cards()

    score -= (
        weights["breaking_stackable_weight"] * board.count_cards_breaking_stackable()
    )

    score -= weights["breaking_sequence_weight"] * board.count_cards_breaking_sequence()

    score += weights["empty_stack_weight"] * board.count_empty_stacks()

    score += weights["sequence_length_weight"] * sum(board.stacks_sequence_lengths())

    score += weights["stacked_length_weight"] * sum(board.stacks_stacked_lengths())

    score += weights["total_rank_sequence_weight"] * board.total_rank_sequence_cards()

    score += weights["total_rank_stacked_weight"] * board.total_rank_stacked_cards()

    score += weights["stacked_length_indicator"] * board.stacked_length_indicator()

    score += weights["sequence_length_indicator"] * board.sequence_length_indicator()

    score += weights["count_completed_stacks"] * board.count_completed_stacks()

    return score


class Move(NamedTuple):
    source_stack: int
    destination_stack: int
    card_index: int


class BFS_element(NamedTuple):
    board: Board
    path: list[Move]


def bfs_first_path(
    initial_board: Board, win_condition, loss_condition=None, max_depth=None
):
    queue = deque([BFS_element(initial_board.clone(), [])])
    visited = set()

    while queue:
        current = queue.popleft()
        current_board, current_path = current.board, current.path
        current_state = current_board.get_hashed_state()

        if current_state in visited:
            continue
        visited.add(current_state)

        if loss_condition and loss_condition(current_board):
            continue

        if win_condition(current_board):
            return current_path

        if max_depth is not None and len(current_path) >= max_depth:
            continue

        for move in current_board.list_available_moves():
            if len(move) == 3:
                move = Move(*move)
                if not current_board.is_move_indifferent(move):
                    continue
                queue.append(_simulate_move(current_board, move, current_path))

    return []


def bfs_all_paths(
    initial_board: Board, win_condition, loss_condition=None, max_depth=None
):
    queue = deque([BFS_element(initial_board.clone(), [])])
    visited = set()
    successful_paths: list[list[Move]] = []

    while queue:
        current = queue.popleft()
        current_board, current_path = current.board, current.path
        current_state = current_board.get_hashed_state()

        if current_state in visited:
            continue
        visited.add(current_state)

        if loss_condition and loss_condition(current_board):
            continue

        if win_condition(current_board):
            successful_paths.append(current_path)
            continue

        # Stop expanding if the maximum depth is reached
        if max_depth is not None and len(current_path) >= max_depth:
            continue

        for move in current_board.list_available_moves():
            if len(move) == 3:
                move = Move(*move)
                queue.append(_simulate_move(current_board, move, current_path))

    return successful_paths


def _simulate_move(board, move, path):
    new_board = board.clone()
    new_board.move_by_index(*move)
    return BFS_element(new_board, path + [move])


def find_progressive_actions(board: Board):
    empty_stacks = board.count_empty_stacks()
    hidden_cards = board.count_hidden_cards()
    visible_cards = board.count_visible_cards()
    initial_completed_stacks = board.count_completed_stacks()
    return bfs_all_paths(
        board,
        lambda board: is_more_empty_stacks(board, empty_stacks)
        or is_fewer_hidden_cards_condition(board, hidden_cards)
        or is_more_visible_cards_condition(board, visible_cards)
        or is_more_completed_stacks(board, initial_completed_stacks),
        max_depth=8,
    )


def find_improved_equivalent_position(board: Board):
    initial_sequence_length = board.sequence_length_indicator()
    initial_completed_stacks = board.count_completed_stacks()

    return bfs_first_path(
        board,
        win_condition=lambda board: is_more_sequence_length_indicator(
            board, initial_sequence_length
        )
        or is_more_completed_stacks(board, initial_completed_stacks),
        max_depth=8,
    )


def find_move_increasing_stacked_length(board: Board):
    initial_sequence_length = board.sequence_length_indicator()
    initial_stacked_length = board.stacked_length_indicator()

    def win_condition_with_logging(board):
        more_stacked = is_more_stacked_length(board, initial_stacked_length)
        less_sequence = is_less_sequence_length_indicator(
            board, initial_sequence_length
        )
        win_condition = more_stacked and not less_sequence
        return win_condition

    return bfs_first_path(
        board,
        win_condition=win_condition_with_logging,
        max_depth=5,
    )


def find_moves_freeing_covered_cards(board: Board):
    empty_stacks = board.count_empty_stacks()
    cards_in_sequence = sum(board.stacks_sequence_lengths())
    breaking_stackable = board.count_cards_breaking_stackable()
    hidden_cards = board.count_hidden_cards()

    return bfs_all_paths(
        board,
        win_condition=lambda board: is_less_cards_breaking_stackable(
            board, breaking_stackable
        )
        and not is_less_cards_in_sequence(board, cards_in_sequence),
        loss_condition=lambda board: is_fewer_hidden_cards_condition(
            board, hidden_cards
        ),
        max_depth=8,
    )


def is_empty_stack_condition(board: Board) -> bool:
    return any(stack.is_empty() for stack in board.stacks)


def is_more_empty_stacks(board: Board, empty_stacks: int) -> bool:
    return board.count_empty_stacks() > empty_stacks


def is_more_completed_stacks(board: Board, completed_stacks: int) -> bool:
    return board.count_completed_stacks() > completed_stacks


def is_fewer_hidden_cards_condition(board: Board, hidden_cards: int) -> bool:
    return board.count_hidden_cards() < hidden_cards


def is_more_visible_cards_condition(board: Board, visible_cards: int) -> bool:
    return board.count_visible_cards() > visible_cards


def is_less_cards_stacked(board: Board, stacked_cards: int) -> bool:
    return sum(board.stacks_stacked_lengths()) < stacked_cards


def is_more_cards_in_sequence(board: Board, sequence_cards: int) -> bool:
    return sum(board.stacks_sequence_lengths()) > sequence_cards


def is_less_cards_in_sequence(board: Board, sequence_cards: int) -> bool:
    return sum(board.stacks_sequence_lengths()) < sequence_cards


def is_less_sequence_length_indicator(board: Board, sequence_length: int) -> bool:
    return board.sequence_length_indicator() < sequence_length


def is_more_sequence_length_indicator(board: Board, sequence_length: int) -> bool:
    return board.sequence_length_indicator() > sequence_length


def is_less_cards_breaking_stackable(board: Board, breaking_stackable: int) -> bool:
    return board.count_cards_breaking_stackable() < breaking_stackable


def is_more_stacked_length(board: Board, stacked_length: int) -> bool:
    return board.stacked_length_indicator() > stacked_length


def is_board_winnable(initial_board: Board):
    queue = deque([BFS_element(initial_board.clone(), [])])
    visited = set()

    while queue:
        current = queue.popleft()
        current_board, current_path = current.board, current.path
        current_state = current_board.get_hashed_state()

        if current_state in visited:
            continue
        visited.add(current_state)

        if current_board.is_game_won():
            return True

        print(len(current_path))

        for move in current_board.list_available_moves():
            if len(move) == 3:
                queue.append(_simulate_move(current_board, move, current_path))
            elif len(move) == 1:
                simulated_board = current_board.clone()
                simulated_board.draw_from_deck()
                queue.append(BFS_element(simulated_board, current_path + [move]))

    return False
