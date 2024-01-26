from __future__ import annotations
from collections import deque
import logging
from typing import NamedTuple, TYPE_CHECKING
from matplotlib.style import available
from deck import Card

if TYPE_CHECKING:
    from spiderSolitaire import Board, Stack


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
    if not search_for_beneficial_reversible_move(board):
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


def search_for_beneficial_reversible_move(board: Board) -> bool:
    """
    Determine if there's a reversible move on the board that could lead to an improved position.
    This function should be called after moving all possible cards that cover an empty stack through reversible moves.

    :param board: The current state of the Spider Solitaire game board.
    :return: True if a reversible move is found that improves the board position, False otherwise.
    """
    amount_of_stacks = len(board.stacks)
    empty_stacks = board.count_empty_stacks()

    for source_index, target_index in _generate_unique_index_pairs(amount_of_stacks):
        source_stack = board.stacks[source_index]
        source_sequences = source_stack.get_accessible_sequences()
        logging.debug(
            f"find_reversible_move: source_sequences = {repr(source_sequences)}"
        )

        target_stack = board.stacks[target_index]
        target_sequences = target_stack.get_accessible_sequences()
        logging.debug(
            f"find_reversible_move: target_sequences = {repr(target_sequences)}"
        )
        if _check_for_reversible_move(
            board,
            source_sequences,
            target_sequences,
            source_index,
            target_index,
            empty_stacks,
        ):
            return True

    return False


def _check_for_reversible_move(
    board, source_sequences, target_sequences, source_index, target_index, empty_stacks
):
    for target_seq_index, target_sequence in enumerate(target_sequences):
        for source_seq_index, source_sequence in enumerate(source_sequences):
            beneficial_merge, beneficial_index = is_beneficial_merge(
                target_sequence[-1], source_sequence
            )
            logging.debug(
                f"find_reversible_move: beneficial_merge = {beneficial_merge}"
            )

            if beneficial_merge:
                merged_sequence = _prepare_merged_sequence(
                    source_sequence,
                    beneficial_index,
                    source_seq_index,
                    source_sequences,
                )
                top_cards = get_uninvolved_top_cards(board, source_index, target_index)
                logging.debug(
                    f"find_reversible_move: merged_sequence = {repr(merged_sequence)}"
                )
                logging.debug(
                    f"find_reversible_move: target_sequences for switch = {repr(target_sequences[(target_seq_index + 1):])}"
                )

                if can_switch_stacked_reversibly(
                    merged_sequence,
                    target_sequences[(target_seq_index + 1) :],
                    top_cards,
                    empty_stacks,
                ):
                    return True
    return False


def _prepare_merged_sequence(
    source_sequence, beneficial_index, source_seq_index, source_sequences
):
    merged_sequence = [source_sequence[beneficial_index:]]
    if source_seq_index + 1 < len(source_sequences):
        merged_sequence.extend(source_sequences[(source_seq_index + 1) :])
    return merged_sequence


def _generate_unique_index_pairs(amount_of_stacks):
    for source_index in range(amount_of_stacks):
        for target_index in range(amount_of_stacks):
            if source_index != target_index:
                yield source_index, target_index


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


def can_move_stacked_reversibly(
    stacked_cards, stacks: list[Stack], empty_stacks=0, strict=False
):
    """
    Determine if stacked card sequences can be moved reversibly given the empty stacks.

    :param stacked_cards: List of lists, each representing a sequence of stacked cards.
    :param stacks: List of Stack objects representing the game's stacks.
    :param empty_stacks: Number of empty stacks available on the board.
    :return: True if the sequences can be moved reversibly, False otherwise.
    """
    top_cards = [stack.top_card() for stack in stacks if not stack.is_empty()]
    degrees_of_freedom = degrees_of_freedom_for_empty_stacks(empty_stacks)
    max_dof_required, used_stacks = dof_to_move_stacked_reversibly(
        stacked_cards, top_cards
    )
    if strict:
        return used_stacks == 0
    return max_dof_required <= degrees_of_freedom


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
    # Validate input parameters
    if not isinstance(empty_stacks, int) or empty_stacks < 0:
        raise ValueError("empty_stacks must be a non-negative integer")
    for stack in [first_stacked, second_stacked]:
        if not all(isinstance(card, Card) for sequence in stack for card in sequence):
            raise ValueError("Stacks must only contain Card objects")

    logging.debug(f"can_switch_staked_reversibly: input top_cards = {repr(top_cards)}")
    degrees_of_freedom = degrees_of_freedom_for_empty_stacks(empty_stacks)
    logging.debug(
        f"can_switch_staked_reversibly: degrees_of_freedom = {degrees_of_freedom}"
    )

    # Handle cases where one of the sequences may be empty
    if not first_stacked or not second_stacked:
        # If one sequence is empty, the operation is simpler and depends on the single non-empty sequence

        logging.debug(
            f"can_switch_staked_reversibly: one of the input stacks is empty = {repr(first_stacked)}, {repr(second_stacked)}"
        )
        non_empty_sequence = first_stacked if first_stacked else second_stacked
        if not non_empty_sequence:
            logging.debug(
                f"can_switch_staked_reversibly: both sequences are empty, return False"
            )
            return False
        dof_needed, used_dof = dof_to_move_stacked_reversibly(
            non_empty_sequence, top_cards
        )

        logging.debug(f"can_switch_staked_reversibly: dof_needed = {dof_needed}")
        # Remove 1 degree of freedom because the other stack can stack the top card of the stacked sequence
        return dof_needed - 1 <= degrees_of_freedom

    # Decide which stack to move first
    first_top_rank = (
        first_stacked[-1][-1].rank
        if first_stacked and first_stacked[-1]
        else float("-inf")
    )
    second_top_rank = (
        second_stacked[-1][-1].rank
        if second_stacked and second_stacked[-1]
        else float("-inf")
    )

    logging.debug(
        f"can_switch_staked_reversibly: top ranked cards = {first_top_rank}, {second_top_rank}"
    )

    if first_top_rank > second_top_rank:
        initial_stack, final_stack = second_stacked, first_stacked
    else:
        initial_stack, final_stack = first_stacked, second_stacked

    dof_needed_initial, used_dof_initial = dof_to_move_stacked_reversibly(
        initial_stack, top_cards
    )

    logging.debug(
        f"can_switch_staked_reversibly: dof_needed_initial = {dof_needed_initial}"
    )

    if dof_needed_initial > degrees_of_freedom:
        logging.debug(
            f"can_switch_staked_reversibly: Not enough DOF to move the first stack"
        )
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
    # -1 degree of freedom because the top sequence can directly be moved on top of the previous stack
    return available_dof >= dof_final_stack - 1


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
    logging.debug(f"dof_to_move_stacked_reversibly: top cards = {repr(top_cards)}")

    if stacked_cards and isinstance(stacked_cards[0], Card):
        stacked_cards = [stacked_cards]

    for sequence in reversed(stacked_cards):
        if not sequence:
            continue
        logging.debug(f"dof_to_move_stacked_reversibly: sequence = {repr(sequence)}")
        if any(card.can_stack(sequence[0]) for card in top_cards):
            current_dof_used = 0
        else:
            partially_movable = any(
                card.can_stack(element)
                for card in top_cards
                for element in sequence[1:]
            )
            current_dof_used = 1 if partially_movable else current_dof_used + 1

        # TODO: DOF used depends on the number of available stacks
        max_dof_used = max(max_dof_used, current_dof_used)

    return max_dof_used, min(1, current_dof_used)


def move_stack_to_temporary_position(
    board: Board,
    stack_id,
    card_index,
    empty_stacks_to_leave,
    ignored_stacks: list[int] = [],
):
    """
    Attempts to move a stack to a temporary position.

    :param stack: The stack of cards to move.
    :param empty_stacks: Total number of available empty stacks.
    :param empty_stacks_to_leave: Number of empty stacks to leave after the move.
    :return: A tuple (bool, list, int), where the first element is a boolean indicating if the move is possible,
             the second is a list of moves, and the third is the number of empty stacks remaining.
    """
    cloned_board = board.clone()
    current_empty_stacks = cloned_board.count_empty_stacks()
    stack_to_move = cloned_board.stacks[stack_id]

    sequences_to_move = stack_to_move.count_sequences_to_index(card_index)
    moves: list[Move] = []

    if sequences_to_move > current_empty_stacks - empty_stacks_to_leave:
        # TODO: Implement more complex behaviors
        return moves

    for sequence_index in range(sequences_to_move):
        # Move each sequence to an empty stack
        # This is a conceptual representation; actual move logic will depend on game's rules
        move = Move(
            stack_id,
            cloned_board.get_empty_stack_id(),
            stack_to_move.first_card_of_valid_sequence(),
        )
        moves.append(move)
        cloned_board.move_by_index(*move)
        current_empty_stacks -= 1

    return moves


def free_stack(board: Board, ignored_stacks: list[int] = []):
    cloned_board = board.clone()
    moves: list[Move] = []
    available_dof = degrees_of_freedom_for_empty_stacks(
        cloned_board.count_empty_stacks()
    )
    initial_empty_stacks = cloned_board.count_empty_stacks()

    stack_to_free_id = _select_stack_to_free(cloned_board, ignored_stacks)
    if stack_to_free_id == -1:
        logging.debug(f"free_stack: No stack can be freed")
        return moves

    print(f"Stack to free = {stack_to_free_id}")

    stack_to_free = cloned_board.stacks[stack_to_free_id]
    target_stack_id = _find_stack_which_can_stack(
        cloned_board, stack_to_free.cards[0], ignored_stacks
    )

    sequences = stack_to_free.get_accessible_sequences()

    # If you can just move to empty stacks, do that as it is easier
    # TODO: This is not the most optimal movement strategy
    length_considered_sequence = 0
    top_cards = get_uninvolved_top_cards(
        cloned_board, stack_to_free_id, stack_to_free_id
    )
    if len(sequences) <= available_dof:
        moves = _move_card_to_no_intermediates(
            cloned_board, stack_to_free_id, target_stack_id, 0
        )
        for move in moves:
            cloned_board.move_by_index(*move)

    else:
        # Find if just moving the sequences a solution is plausible
        # If a solution is not plausible, you need to split the sequences
        moves_no_split: list[Move] = []
        can_move_without_splitting = True
        testing_board = board.clone()
        for current_seq_index, sequence in enumerate(reversed(sequences)):
            print("Cycling")
            length_considered_sequence += 1
            temporary_stack_id = _find_stack_which_can_stack(
                testing_board, sequence[0], ignored_stacks
            )
            move_index = stack_to_free.cards.index(sequence[0])
            if temporary_stack_id != -1:
                partial_moves = _move_card_to_no_intermediates(
                    testing_board, stack_to_free_id, temporary_stack_id, move_index
                )
                if partial_moves:
                    length_considered_sequence = 0
                    moves_no_split.extend(partial_moves)
                    for move in partial_moves:
                        testing_board.move_by_index(*move)

                    if len(sequences) - current_seq_index <= available_dof:
                        break

                    while testing_board.count_empty_stacks() < initial_empty_stacks:
                        more_moves = free_stack(testing_board, ignored_stacks)
                        moves_no_split.extend(more_moves)
                        for move in more_moves:
                            testing_board.move_by_index(*move)
                else:
                    can_move_without_splitting = False
                    break

            if length_considered_sequence > available_dof:
                can_move_without_splitting = False
                break

        if can_move_without_splitting:
            cloned_board = testing_board
            moves = moves_no_split
        else:
            # TODO: Find advantageous splitting sequence so that you can keep organizing
            return []

    while cloned_board.count_empty_stacks() <= initial_empty_stacks:
        more_moves = free_stack(cloned_board, ignored_stacks)
        moves.extend(more_moves)
        for move in more_moves:
            cloned_board.move_by_index(*move)

    return moves


def _move_card_to_no_intermediates(
    board: Board, source_id, target_id, card_to_move
) -> list[Move]:
    """
    Move a card to another stack reversibly, only if allowed by DoF directly with no intermediate exchanges.

    :param board: The current state of the Spider Solitaire game.
    :param source_id: ID of the source stack.
    :param target_id: ID of the target stack.
    :param card_id: ID of the card in the source stack to start moving from.
    :return: List of Moves needed to perform the action.
    """
    moves: list[Move] = []
    cloned_board = board.clone()
    available_dof = degrees_of_freedom_for_empty_stacks(
        cloned_board.count_empty_stacks()
    )
    source_stack = cloned_board.stacks[source_id]
    sequences = cards_to_sequences(source_stack.cards[card_to_move:])
    logging.debug(f"_move_card_to_no_intermediates: sequences = {sequences}")

    if not source_stack.is_stacked(card_to_move) or len(sequences) - 1 > available_dof:
        logging.debug(
            f"_move_card_to_no_intermediates: Returning early as the input is invalid"
        )
        return moves

    if len(sequences) > 1:
        stack_to_stack_moves = optimal_stacked_reversible_movement(
            cloned_board, source_id, len(sequences) - 1
        )
        for move_set in stack_to_stack_moves:
            start, dest = move_set
            card_id = cloned_board.stacks[start].first_card_of_valid_sequence()
            move = Move(start, dest, card_id)
            cloned_board.move_by_index(*move)
            moves.append(move)

    move = Move(source_id, target_id, card_to_move)
    moves.append(move)
    logging.debug(f"_move_card_to_no_intermediates: moves = {moves}")
    return moves


def optimal_stacked_reversible_movement(
    board: Board,
    source_stack_id: int,
    amount_of_sequences: int,
):
    """
    Generates an optimal sequence of moves to free one stack given a number of empty stacks,
    ensuring that the number of moves starting from the source stack matches the amount of sequences.
    It does not consider moving cads on top of others, it only uses the empty stacks.

    :param board: The current state of the Spider Solitaire game.
    :param source_stack_id: ID of the source stack from which cards are to be moved.
    :param amount_of_sequences: Number of sequences in the source stack to be moved.
    :return: List of tuples representing the moves, where each tuple is (source_stack_id, target_stack_id).
    """
    initial_empty_stacks = board.count_empty_stacks()
    empty_stacks = [id for id, stack in enumerate(board.stacks) if stack.is_empty()]
    moves: list[tuple[int, int]] = []

    if initial_empty_stacks == 0:
        return moves
    if initial_empty_stacks == 1:
        moves.append((source_stack_id, empty_stacks[0]))
    if initial_empty_stacks == 2:
        moves.append((source_stack_id, empty_stacks[0]))
        if amount_of_sequences == 1:
            return moves
        moves.append((source_stack_id, empty_stacks[1]))
        if amount_of_sequences == 2:
            return moves
        moves.append((empty_stacks[0], empty_stacks[1]))
        moves.append((source_stack_id, empty_stacks[0]))
    if initial_empty_stacks >= 3:
        moves.append((source_stack_id, empty_stacks[0]))
        if amount_of_sequences == 1:
            return moves
        moves.append((source_stack_id, empty_stacks[1]))
        if amount_of_sequences == 2:
            return moves
        moves.append((source_stack_id, empty_stacks[2]))
        if amount_of_sequences == 3:
            return moves
        moves.append((empty_stacks[0], empty_stacks[1]))
        moves.append((source_stack_id, empty_stacks[0]))
        if amount_of_sequences == 4:
            return moves
        moves.append((empty_stacks[2], empty_stacks[0]))
        moves.append((empty_stacks[1], empty_stacks[2]))
        moves.append((empty_stacks[1], empty_stacks[0]))
        moves.append((source_stack_id, empty_stacks[1]))
        if amount_of_sequences == 5:
            return moves
        moves.append((empty_stacks[2], empty_stacks[0]))
        moves.append((source_stack_id, empty_stacks[2]))
        if amount_of_sequences == 6:
            return moves
        moves.append((empty_stacks[1], empty_stacks[2]))
        moves.append((source_stack_id, empty_stacks[1]))

    return moves


def cards_to_sequences(cards: list[Card]) -> list[list[Card]]:
    """Takes a list of cards and returns a list of sequences"""
    sequences: list[list[Card]] = []
    if not cards:
        return sequences
    current_sequence: list[Card] = [cards[0]]
    for card in cards[1:]:
        if current_sequence[-1].can_sequence(card):
            current_sequence.append(card)
        else:
            sequences.append(current_sequence)
            current_sequence = [card]
    if current_sequence:
        sequences.append(current_sequence)

    return sequences


def _select_stack_to_free(board: Board, ignored_stacks: list[int]) -> int:
    highest_rank = -1
    selected_stack_id = -1
    available_dof = degrees_of_freedom_for_empty_stacks(board.count_empty_stacks())
    for id, stack in enumerate(board.stacks):
        if id in ignored_stacks or stack.is_empty() or not stack.is_stacked_on_table():
            continue

        rank = stack.cards[0].rank
        if rank <= highest_rank or rank == 14:
            continue

        cards_to_move = stack.cards
        if not cards_to_move:
            continue

        stacks_to_consider = [
            cid
            for cid, other_stack in enumerate(board.stacks)
            if cid != id and not cid in ignored_stacks and not other_stack.is_empty()
        ]
        for dest in stacks_to_consider:
            if not board.stacks[dest].can_stack(cards_to_move[0]):
                continue

            dof_needed, _ = dof_to_move_stacked_reversibly(
                cards_to_sequences(cards_to_move),
                get_uninvolved_top_cards(board, id, id),
            )

            if dof_needed <= available_dof:
                highest_rank = rank
                selected_stack_id = id
    return selected_stack_id


def _find_stack_which_can_stack(
    board: Board, card: Card, ignored_stacks: list[int] = []
) -> int:
    target_stack = -1

    for id, stack in enumerate(board.stacks):
        if not id in ignored_stacks and not stack.is_empty():
            if stack.can_stack(card):
                target_stack = id
                break

    return target_stack


def find_stack_to_move_sequence(
    board: Board, top_card: Card, ignored_stacks: list[int] = [], ignore_empty=False
):
    """
    Find a stack to which the sequence can be legally moved.

    :param board: The current board state.
    :param sequence: The sequence of cards to move.
    :return: ID of the stack where the sequence can be moved, or None if not found.
    """
    for target_stack_id, target_stack in enumerate(board.stacks):
        if target_stack_id in ignored_stacks:
            continue
        if ignore_empty and target_stack.is_empty():
            continue
        if target_stack.can_stack(top_card):
            return target_stack_id
    return None


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
