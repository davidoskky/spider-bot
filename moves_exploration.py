from __future__ import annotations
from collections import deque
from typing import NamedTuple, TYPE_CHECKING
from matplotlib.style import available

if TYPE_CHECKING:
    from spiderSolitaire import Board, Stack
    from deck import Card


DEFAULT_WEIGHTS = {
    "visible_card_weight": 10,
    "hidden_card_weight": 10,
    "breaking_stackable_weight": 69,
    "breaking_sequence_weight": 138,
    "empty_stack_weight": 4000,
    "semi_empty_stacks": 4000,
    "count_blocked_stacks": -500,
    "count_completed_stacks": 100000,
    "count_accessible_full_sequences": 4000,
    "sequence_length_weight": 200,
    "stacked_length_weight": 65,
    "total_rank_sequence_weight": 40,
    "total_rank_stacked_weight": 20,
    "stacked_length_indicator": 2,
    "sequence_length_indicator": 5,
}


def score_board(board: Board, weights) -> int:
    """
    Evaluate and score the board state based on various criteria.

    :param board: The current state of the Spider Solitaire game.
    :param weights: A dictionary of weights for different scoring criteria.
    :return: The score of the board.
    """
    score: int = 0

    visible_card_score = weights["visible_card_weight"] * board.count_visible_cards()
    score += visible_card_score

    hidden_card_score = -weights["hidden_card_weight"] * board.count_hidden_cards()
    score += hidden_card_score

    breaking_stackable_score = (
        -weights["breaking_stackable_weight"] * board.count_cards_breaking_stackable()
    )
    score += breaking_stackable_score

    breaking_sequence_score = (
        -weights["breaking_sequence_weight"] * board.count_cards_breaking_sequence()
    )
    score += breaking_sequence_score

    empty_stack_score = weights["empty_stack_weight"] * board.count_empty_stacks()
    score += empty_stack_score

    accessible_full_sequences_score = (
        weights["count_accessible_full_sequences"]
        * board.count_accessible_full_sequences()
    )
    score += accessible_full_sequences_score

    semi_empty_stack_score = (
        weights["semi_empty_stacks"] * board.count_semi_empty_stacks()
    )
    score += semi_empty_stack_score

    blocked_stacks_score = (
        weights["count_blocked_stacks"] * board.count_blocked_stacks()
    )
    score += blocked_stacks_score

    sequence_length_score = weights["sequence_length_weight"] * sum(
        board.stacks_sequence_lengths()
    )
    score += sequence_length_score

    stacked_length_score = weights["stacked_length_weight"] * sum(
        board.stacks_stacked_lengths()
    )
    score += stacked_length_score

    total_rank_sequence_score = (
        weights["total_rank_sequence_weight"] * board.total_rank_sequence_cards()
    )
    score += total_rank_sequence_score

    total_rank_stacked_score = (
        weights["total_rank_stacked_weight"] * board.total_rank_stacked_cards()
    )
    score += total_rank_stacked_score

    stacked_length_indicator_score = (
        weights["stacked_length_indicator"] * board.stacked_length_indicator()
    )
    score += stacked_length_indicator_score

    sequence_length_indicator_score = (
        weights["sequence_length_indicator"] * board.sequence_length_indicator()
    )
    score += sequence_length_indicator_score

    completed_stacks_score = (
        weights["count_completed_stacks"] * board.count_completed_stacks()
    )
    score += completed_stacks_score

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

        # Use only moves indifferent to the board state which don't change freedom
        for move in current_board.list_indifferent_moves():
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
        max_depth=6,
    )


def find_improved_equivalent_position(board: Board):
    if not find_reversible_move(board):
        return []

    initial_sequence_length = board.sequence_length_indicator()
    initial_completed_stacks = board.count_completed_stacks()

    return bfs_first_path(
        board,
        win_condition=lambda board: is_more_sequence_length_indicator(
            board, initial_sequence_length
        )
        or is_more_completed_stacks(board, initial_completed_stacks),
        max_depth=6,
    )


# TODO: This is not correct it should be sequence length can be increased and consider
# like 9 - 8 - 7 - 6 and 6 - 5 - 4 - 3 - 2 - 1 can still be merged even though 6 cannot sequence 6
def _identify_plausible_improved_equivalent_positions(board: Board) -> bool:
    """
    Determine if there are any plausible moves on the board that could lead to an improved position.

    This function analyzes each stack on the board, identifying the initial and final ranks
    of the sequences within each stack. It then checks if a card from one sequence (based on its rank)
    can be moved to the end of another sequence. This is a potential move that could lead to an
    improved board position.

    :param board: The current state of the Spider Solitaire game board.
    :return: True if there is at least one plausible move that could improve the board position, False otherwise.
    """
    potential_movable_cards = set()
    potential_destination_cards = set()

    for stack in board.stacks:
        sequences = stack.get_accessible_sequences()

        if sequences:
            if stack.first_accessible_sequence == 0:
                potential_movable_cards.add(sequences[0][0])

            for i, sequence in enumerate(sequences):
                if i != 0:
                    potential_movable_cards.add(sequence[0])
                potential_destination_cards.add(sequence[-1])

    return any(
        destination.can_sequence(movable)
        for destination in potential_destination_cards
        for movable in potential_destination_cards
    )


def find_reversible_move(board: Board) -> bool:
    """
    Determine if there's a reversible move on the board that could lead to an improved position.
    This function should be called after moving all possible cards that cover an empty stack through reversible moves.

    :param board: The current state of the Spider Solitaire game board.
    :return: True if a reversible move is found that improves the board position, False otherwise.
    """
    amount_of_stacks = len(board.stacks)
    empty_stacks = board.count_empty_stacks()

    for source_index in range(amount_of_stacks):
        source_stack = board.stacks[source_index]
        source_sequences = source_stack.get_accessible_sequences()

        for target_index in range(amount_of_stacks):
            if source_index == target_index:
                continue

            target_stack = board.stacks[target_index]
            target_sequences = target_stack.get_accessible_sequences()

            for target_seq_index, target_sequence in enumerate(target_sequences):
                for source_seq_index, source_sequence in enumerate(source_sequences):
                    beneficial_merge, beneficial_index = is_beneficial_merge(
                        target_sequence[-1], source_sequence
                    )
                    if beneficial_merge:
                        # Prepare the merged sequence
                        merged_sequence = [source_sequence[beneficial_index + 1 :]]
                        if source_seq_index + 1 < len(source_sequences):
                            merged_sequence.extend(
                                source_sequences[(source_seq_index + 1) :]
                            )

                        top_cards = get_uninvolved_top_cards(
                            board, source_index, target_index
                        )

                        if can_switch_stacked_reversibly(
                            merged_sequence,
                            target_sequences[target_seq_index + 1 :],
                            top_cards,
                            empty_stacks,
                        ):
                            return True

    return False


def get_uninvolved_top_cards(board: Board, source_index: int, target_index: int):
    return [
        stack.top_card()
        for i, stack in enumerate(board.stacks)
        if i not in [source_index, target_index] and not stack.is_empty()
    ]


def is_beneficial_merge(
    target_card: Card, source_sequence: list[Card]
) -> tuple[bool, int]:
    """
    Determine if merging the target sequence with the source sequence is beneficial.

    :param target_sequence: The sequence where the merge would end.
    :param source_sequence: The sequence to merge into the target sequence.
    :return: True if merging is beneficial, False otherwise.
    """
    # Check if the last card of the target sequence and the first card of the source sequence have the same rank
    if target_card.rank != source_sequence[0].rank:
        return False, 0

    for i, source_card in enumerate(source_sequence):
        if target_card.can_sequence(source_card):
            return True, i

    return False, 0


def find_move_increasing_stacked_length(board: Board):
    # if not _identify_plausible_increasing_stacked(board):
    #    return []

    initial_sequence_length = board.sequence_length_indicator()
    initial_stacked_length = board.stacked_length_indicator()
    initial_empty_stacks = board.count_empty_stacks()

    def win_condition_with_logging(board):
        more_stacked = is_more_stacked_length(board, initial_stacked_length)
        less_sequence = is_less_sequence_length_indicator(
            board, initial_sequence_length
        )
        more_empty = is_more_empty_stacks(board, initial_empty_stacks)
        win_condition = (more_stacked or more_empty) and not less_sequence
        return win_condition

    return bfs_first_path(
        board,
        win_condition=win_condition_with_logging,
        max_depth=5,
    )


# TODO: Improve this as well
def _identify_plausible_increasing_stacked(board: Board) -> bool:
    """
    Determine if there are any plausible moves on the board that could lead to an improved position.

    This function analyzes each stack on the board, identifying the initial and final ranks
    of the sequences within each stack. It then checks if a card from one sequence (based on its rank)
    can be moved to the end of another sequence. This is a potential move that could lead to an
    improved board position.

    :param board: The current state of the Spider Solitaire game board.
    :return: True if there is at least one plausible move that could improve the board position, False otherwise.
    """
    potential_movable_cards = set()
    potential_destination_cards = set()

    for stack in board.stacks:
        sequences = stack.get_accessible_sequences()

        if sequences:
            if stack.first_accessible_sequence == 0:
                potential_movable_cards.add(sequences[0][0])

            for i, sequence in enumerate(sequences):
                if i != 0:
                    potential_movable_cards.add(sequence[0])
                potential_destination_cards.add(sequence[-1])

    return any(
        destination.can_stack(movable)
        for destination in potential_destination_cards
        for movable in potential_destination_cards
    )


def can_uncover_specific_card(
    board: Board, target_stack_index: int, card_position: int
) -> bool:
    """
    Determine if it is possible to uncover a specific hidden card without uncovering
    other hidden cards or drawing from the deck.

    :param board: The current state of the Spider Solitaire game board.
    :param target_stack_index: The index of the stack where the target card is located.
    :param card_position: The position of the card in the stack (0 being the top).
    :return: True if the card can be uncovered, False otherwise.
    """
    target_stack = board.stacks[target_stack_index]

    # Check if the target card is already visible
    if card_position != target_stack.first_visible_card - 1:
        return False

    # Analyze cards above the target card
    visible_sequences = target_stack.visible_cards()
    if not _can_move_card(board, card_position + 1, target_stack_index):
        return False

    return True


def _can_move_card(board: Board, card, current_stack_index) -> bool:
    """
    Check if a specific card can be moved to another stack.

    :param board: The current state of the board.
    :param card: The card to check.
    :param current_stack_index: The index of the stack where the card currently is.
    :return: True if the card can be moved, False otherwise.
    """
    current_stack = board.stacks[current_stack_index]
    if board.count_empty_stacks() == 0:
        if card == len(current_stack.cards):
            for stack in board.stacks:
                if stack != current_stack and stack.can_stack(
                    current_stack.cards[card]
                ):
                    return True
        return False

    degrees_of_freedom = pow(2, board.count_empty_stacks()) - 1
    max_stacked_sequences = (degrees_of_freedom + 1) / 2
    return True


def can_move_stacked_reversibly(stacked_cards, stacks, empty_stacks=0):
    """
    Determine if stacked card sequences can be moved reversibly given the empty stacks.

    :param stacked_cards: List of lists, each representing a sequence of stacked cards.
    :param stacks: List of Stack objects representing the game's stacks.
    :param empty_stacks: Number of empty stacks available on the board.
    :return: True if the sequences can be moved reversibly, False otherwise.
    """
    top_cards = [stack.top_card for stack in stacks if not stack.is_empty()]
    degrees_of_freedom = degrees_of_freedom_for_empty_stacks(empty_stacks)
    return (
        dof_to_move_stacked_reversibly(stacked_cards, top_cards)[0]
        <= degrees_of_freedom
    )


def degrees_of_freedom_for_empty_stacks(empty_stacks: int) -> int:
    """Calculate degrees of freedom based on empty stacks."""
    if empty_stacks < 0:
        return 0
    return pow(2, empty_stacks) - 1


def can_switch_stacked_reversibly(
    first_stacked: list[list[Card]],
    second_stacked: list[list[Card]],
    top_cards: list[Card],
    empty_stacks: int = 0,
) -> bool:
    """
    Determine if two stacked sequences can switch positions.

    :param first_stacked: First sequence of stacked cards.
    :param second_stacked: Second sequence of stacked cards.
    :param stacks: Game's stacks.
    :param empty_stacks: Available empty stacks.
    :return: True if the sequences can switch positions, else False.
    """
    degrees_of_freedom = degrees_of_freedom_for_empty_stacks(empty_stacks)

    # Handle cases where one of the sequences may be empty
    if not first_stacked or not second_stacked:
        # If one sequence is empty, the operation is simpler and depends on the single non-empty sequence
        non_empty_sequence = first_stacked if first_stacked else second_stacked
        dof_needed, used_dof = dof_to_move_stacked_reversibly(
            non_empty_sequence, top_cards
        )
        return dof_needed <= degrees_of_freedom

    # Determine which stack to move first based on the rank of the top card
    if first_stacked[-1][-1].rank > second_stacked[-1][-1].rank:
        initial_stack, final_stack = second_stacked, first_stacked
    else:
        initial_stack, final_stack = first_stacked, second_stacked

    dof_needed_initial, used_dof_initial = dof_to_move_stacked_reversibly(
        initial_stack, top_cards
    )
    if dof_needed_initial > degrees_of_freedom:
        return False

    # Update available degrees of freedom
    while used_dof_initial > 0:
        # At most 2^N/2 stacked sequences on the most full empty stack
        max_stacked = pow(2, empty_stacks) / 2
        empty_stacks -= 1
        used_dof_initial -= max_stacked

    available_dof = degrees_of_freedom_for_empty_stacks(empty_stacks)

    top_cards.append(initial_stack[-1][-1])
    dof_final_stack, _ = dof_to_move_stacked_reversibly(final_stack, top_cards)
    return available_dof >= dof_final_stack


def dof_to_move_stacked_reversibly(
    stacked_cards: list[list[Card]], top_cards: list[Card]
):
    """
    Calculate the degrees of freedom required to move stacked card sequences reversibly.

    :param stacked_cards: List of lists, each representing a sequence of stacked cards.
    :param top_cards: List of the accessible top cards
    :return: The minimum degrees of freedom required to move the sequences.
    """
    max_dof_used = 0
    current_dof_used = 0

    for sequence in reversed(stacked_cards):
        if any(card.can_stack(sequence[0]) for card in top_cards if sequence):
            current_dof_used = 0
        else:
            partially_movable = any(
                card.can_stack(element)
                for card in top_cards
                for element in sequence[1:]
            )
            current_dof_used = 1 if partially_movable else current_dof_used + 1

        max_dof_used = max(max_dof_used, current_dof_used)

    return max_dof_used, current_dof_used


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
        max_depth=4,
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
