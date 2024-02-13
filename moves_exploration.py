from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, NamedTuple

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


def find_improved_equivalent_position_manual(board: Board) -> list[Move]:
    """
    Finds a set of reversible moves leading to an improved equivalent position for a card within the board by moving it to a different stack
    where it extends a sequence, enhancing the board's overall state.

    Parameters:
    - board (Board): The current state of the board.

    Returns:
    - List[Move]: A list of moves that result in an improved board position, or an empty list if no such move exists.
    """
    for source_stack_index, source_stack in enumerate(board.stacks):
        if source_stack.is_empty():
            continue

        # Consider the first card if it's the only card, otherwise consider only the first card in sequence
        # TODO: What if the first stacked card is in sequence with the rest? This doesn't work
        cards_to_consider = (
            source_stack.cards
            if source_stack.first_card_of_valid_stacked() == 0
            else source_stack.get_stacked()[1:]
        )
        logging.debug(
            f"find_improved_equivalent_position_manual: cards_to_consider = {cards_to_consider}"
        )

        for card in cards_to_consider:
            if card.rank == 14: # The king cannot be moved reversibly
                continue
            card_index = (
                len(source_stack.cards) - 1 - source_stack.cards[::-1].index(card)
            )

            for target_stack_index, target_stack in enumerate(board.stacks):
                # Don't consider moving to empty stack, this check should not be necessary, but just in case
                if target_stack_index == source_stack_index or target_stack.is_empty():
                    continue

                target_card_index = find_placement_index_for_card(
                    card, target_stack, should_sequence=True
                )
                if target_card_index is None:
                    continue

                logging.debug(
                    f"find_improved_equivalent_position_manual: target_card_ind = {target_card_index}"
                )

                target_card = target_stack.cards[target_card_index]
                if is_sequence_improved(source_stack, target_stack, card, target_card):
                    logging.debug(
                        f"find_improved_equivalent_position_manual: source = {source_stack_index}, target = {target_stack_index}, card = {card_index}"
                    )
                    moves = move_cards_removing_interfering(
                        board, source_stack_index, target_stack_index, card_index
                    )
                    if moves:
                        return moves

    return []


def is_sequence_improved(
    source_stack: Stack, target_stack: Stack, source_card: Card, target_card: Card
) -> bool:
    """
    Determines if moving a card to a target index in a different stack results in a longer sequence by merging
    accessible sequences from the source and target stacks.

    :param source_stack: The stack from which the card is moved.
    :param target_stack: The stack to which the card is moved.
    :param card: The card being moved.
    :param target_index: The target position in the target stack.
    :return: True if the sequence is improved, False otherwise.
    """

    source_sequences = source_stack.get_accessible_sequences()
    target_sequences = target_stack.get_accessible_sequences()

    # Find the sequence in the source stack that includes the card
    source_sequence = None
    for seq in source_sequences:
        if source_card in seq:
            source_sequence = seq
            break

    # If the card is not in any accessible sequence or there's no sequence in the source, no improvement can be made
    if not source_sequence:
        return False

    # Find the target sequence in the target stack that would merge with the source sequence
    target_sequence = []
    for seq in target_sequences:
        if target_card in seq:
            target_sequence = seq
            break

    if not target_sequence:
        return False

    source_pos_from_end = len(source_sequence) - source_sequence.index(source_card)
    target_pos_from_start = target_sequence.index(target_card) + 1

    combined_length = source_pos_from_end + target_pos_from_start

    return combined_length > len(target_sequence) and combined_length > len(
        source_sequence
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


def move_cards_removing_interfering(
    board: Board,
    source_stack_id: int,
    target_stack_id: int,
    source_card_id: int,
) -> list[Move]:
    """
    Relocates a card from a source stack to a destination stack within the game board. If the destination stack has
    interfering cards on top of the intended placement position, those cards are temporarily moved to a different stack
    to allow the relocation. The process only uses reversible moves.

    Steps:
    1. Determine the placement position for the card in the destination stack.
    2. Temporarily relocate any interfering cards at the destination position to another stack.
    3. Move the specified card to the identified position in the destination stack.

    Parameters:
    - game_board (Board): The current state of the game board.
    - from_stack_index (int): The index of the stack from which the card is being moved.
    - to_stack_index (int): The index of the stack to which the card is being moved.
    - card_index (int): The index of the card within the source stack that is to be moved.

    Returns:
    - list[Move]: A list of Move objects representing the required actions to relocate the card, including any temporary
      moves to clear interfering cards.
    """
    moves: list[Move] = []
    cloned_board = board.clone()

    target_stack = cloned_board.get_stack(target_stack_id)
    source_stack = cloned_board.get_stack(source_stack_id)
    moving_card = source_stack.cards[source_card_id]

    target_card_id = find_placement_index_for_card(moving_card, target_stack)
    logging.debug(f"move_cards_removing_interfering: target_card_id = {target_card_id}")
    # Check if the card can be placed in the target stack
    if target_card_id is None:
        return moves

    # Clear any cards covering the target position
    # Maybe this should be done by sequence as well
    clearance_moves = _move_stacked_to_temporary_position(
        cloned_board, target_stack_id, target_card_id + 1
    )
    logging.debug(f"move_cards_removing_interfering: clearance_moves = {clearance_moves}")
    cloned_board.execute_moves(clearance_moves)

    moves_to_complete_switch = move_card_to_top(
        cloned_board, source_stack_id, target_stack_id, source_card_id
    )
    if moves_to_complete_switch:
        moves.extend(clearance_moves)
        moves.extend(moves_to_complete_switch)
    else:
        cloned_board = board.clone()
        clearance_moves = _move_stacked_to_temporary_position(
            cloned_board, source_stack_id, source_card_id + 1,ignored_stacks=[target_stack_id]
        )
        cloned_board.execute_moves(clearance_moves)
        second_clearance_moves = _move_stacked_to_temporary_position(
            cloned_board, target_stack_id, target_card_id + 1
        )
        cloned_board.execute_moves(second_clearance_moves)
        moves_to_complete_switch = move_card_to_top(
            cloned_board, source_stack_id, target_stack_id, source_card_id
        )
        if moves_to_complete_switch:
            moves.extend(clearance_moves)
            moves.extend(second_clearance_moves)
            moves.extend(moves_to_complete_switch)

    return moves


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
                top_cards = get_top_cards_board(board, [source_index, target_index])
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


def get_top_cards_board(board: Board, ignored_stacks: list[int]) -> list[Card]:
    """
    Retrieves the top card from each stack in the board, excluding specified stacks and empty ones.

    Parameters:
    - board (Board): The current state of the board.
    - ignored_stacks (List[int]): Indices of stacks to be ignored.

    Returns:
    - List[Card]: A list of top cards from the stacks not ignored and not empty.
    """
    filtered_stacks = [stack for i, stack in enumerate(board.stacks) if i not in ignored_stacks]

    return get_top_cards(filtered_stacks)

def get_top_cards(stacks: list[Stack]):
    """
    Retrieves the top card from each given stack, excluding empty ones.

    Parameters:
    - stacks (List[Stack]): A list of stacks to retrieve top cards from.

    Returns:
    - List[Card]: A list of top cards from non-empty stacks. Excludes None values.
    """
    return [
        card
        for stack in stacks
        if not stack.is_empty() and (card := stack.top_card()) is not None
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
    top_cards = get_top_cards(stacks)
    degrees_of_freedom = degrees_of_freedom_for_empty_stacks(empty_stacks)
    max_dof_required, used_stacks = dof_to_move_stacked_reversibly(
        stacked_cards, top_cards
    )
    if strict:
        return used_stacks == 0
    return max_dof_required <= degrees_of_freedom


def dof_board(board: Board):
    return degrees_of_freedom_for_empty_stacks(board.count_empty_stacks())


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
        dof_needed, _ = dof_to_move_stacked_reversibly(non_empty_sequence, top_cards)

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



def _move_stacked_to_temporary_position(
        board: Board, stack_id: int, card_index: int, ignored_stacks: list[int] = []
) -> list[Move]:
    """
    Moves a card, along with any cards stacked on top of it, from the specified stack to a temporary stack.
    This function is useful for clearing space or reorganizing cards within the game board.

    The function identifies a suitable temporary stack (if available) and moves the specified card and all cards above it
    within the same stack to this temporary location. It is designed to assist in complex card movements where
    intermediate steps are required to achieve the desired board configuration.

    Parameters:
    - board (Board): The current state of the game board, encapsulating all stacks and their cards.
    - stack_id (int): The index of the stack from which the card and any cards above it are to be moved.
    - card_index (int): The index of the card within the stack from which the movement begins. All cards above this
      index, including the card at this index, are considered part of the move.

    Returns:
    - list[Move]: A list of 'Move' objects representing the steps required to relocate the specified card and any
      cards stacked on top of it to a temporary stack. An empty list indicates that no suitable temporary stack
      was found or that the movement is not possible due to game rules or board state.

    Note:
    - The function assumes that the board, stack IDs, and card indices are valid and does not perform extensive
      validation checks.
    - The choice of the temporary stack is determined by an internal strategy that may vary based on the board's
      current state and the specific rules or constraints of the game being played.
    """
    logging.debug(f"_move_stacked_to_temporary_position: attempting to move card {card_index} from stack {stack_id}")
    moves = []
    clearing_moves = []
    cloned_board = board.clone()
    stack = cloned_board.get_stack(stack_id)
    if card_index >= len(stack.cards):
        logging.debug(f"_move_stacked_to_temporary_position: card index out of bounds")
        return []

    temporary_stack_ids = find_stacks_to_move_card(
        cloned_board, stack.cards[card_index], ignored_stacks=[], ignore_empty=True
    )
    if not temporary_stack_ids:
        if cloned_board.count_empty_stacks() == 0:
            clearing_moves = free_stack(cloned_board)
            if clearing_moves:
                cloned_board.execute_moves(clearing_moves)
            else:
                return []
        temporary_stack_ids = find_stacks_to_move_card(
            cloned_board, stack.cards[card_index], ignored_stacks=[], ignore_empty=False
        )

    logging.debug(f"_move_stacked_to_temporary_position: plausible stacks = {temporary_stack_ids}")

    for temporary_stack_id in temporary_stack_ids:
        if temporary_stack_id in ignored_stacks:
            continue
        logging.debug(f"_move_stacked_to_temporary_position: stack = {stack_id}, card = {card_index}, temporary = {temporary_stack_id}")
        moves = move_card_to_top(cloned_board, stack_id, temporary_stack_id, card_index)
        if moves:
            break

    if clearing_moves:
        clearing_moves.extend(moves)
        moves = clearing_moves
     
    logging.debug(f"_move_stacked_to_temporary_position: moves = {moves}")

    return moves

def free_stack(board: Board, ignored_stacks: list[int] = []) -> list[Move]:
    moves: list[Move] = []
    cloned_board = board.clone()
    initial_empty_stacks = cloned_board.count_empty_stacks()

    stack_to_free_id = _select_stack_to_free(cloned_board, ignored_stacks)
    if stack_to_free_id is None:
        logging.debug(f"free_stack: No stack can be freed")
        return moves

    logging.debug(f"Stack to free = {stack_to_free_id}")

    stack_to_free = cloned_board.stacks[stack_to_free_id]
    target_stack_id = find_stack_to_move_sequence(
        cloned_board, stack_to_free.cards[0], ignored_stacks, ignore_empty=True
    )

    if target_stack_id is None:
        logging.debug("free_stack: No target stack found, cannot move the sequence")
        return moves

    moves = move_card_to_top(board, stack_to_free_id, target_stack_id, 0)

    logging.debug(f"free_stack: moves = {moves}")
    if not moves:
        ignored_stacks.append(stack_to_free_id)
        return free_stack(board, ignored_stacks = ignored_stacks)

    cloned_board.execute_moves(moves)

    while cloned_board.count_empty_stacks() <= initial_empty_stacks:
        more_moves = free_stack(cloned_board, ignored_stacks)
        if not more_moves:
            return []
        moves.extend(more_moves)
        cloned_board.execute_moves(more_moves)

    return moves


def stacks_which_can_be_freed(board: Board) -> int:
    """
    Determines how many stacks can be freed on the board through reversible moves.

    Parameters:
    - board (Board): The current state of the board.

    Returns:
    - int: The number of additional stacks that can be freed through reversible moves.
    """
    cloned_board = board.clone()
    initial_free_stacks = cloned_board.count_empty_stacks()

    moves = free_stack(board)
    while moves:
        board.execute_moves(moves)
        moves = free_stack(board)

    final_free_stacks = cloned_board.count_empty_stacks()

    return final_free_stacks - initial_free_stacks


def _reversible_move_away_from_stack(
    board: Board,
    stack_to_free_id: int,
    card_id,
    ignored_stacks: list[int],
) -> list[Move]:
    """
    Attempts to find and execute a reversible move to transfer a sequence of cards from a specific stack to another,
    aiming to free up the stack while considering the current degrees of freedom (DoF) on the board. The function
    can optionally split the sequence if it leads to a reversible move and respects the game's constraints.
    """
    logging.debug(f"_reversible_move_away_from_stack: Attempting reversible move from stack {stack_to_free_id} starting from card {card_id}")
    direct_moves = _move_away_direct(board, stack_to_free_id, card_id, ignored_stacks)
    if direct_moves:
        return direct_moves
    return _move_away_splitting(board, stack_to_free_id, card_id, ignored_stacks)

def _move_away_splitting(board: Board, stack_id: int, card_id: int, ignored_stacks: list[int] = []) ->list[Move]:
    logging.debug(f"_move_away_splitting: Attempting to split and move away cards from stack {stack_id} starting from card {card_id}")
    ignored_stacks.append(stack_id)

    for considered_card_id in range(card_id, len(board.get_stack(stack_id).cards)):
        can_move = _move_away_direct(board, stack_id, considered_card_id, ignored_stacks)
        if can_move:
            return can_move
    return []


def _move_away_direct(board: Board, stack_id: int, card_id: int, ignored_stacks: list[int] = []) -> list[Move]:
    logging.debug(f"_move_away_direct: Attempting direct move for card {card_id} in stack {stack_id}")
    target_stack_id = find_stack_to_move_sequence(
                board, board.get_card(stack_id, card_id), ignored_stacks, ignore_empty=False
            )
    if target_stack_id is not None:
        return _move_card_to_no_splits(board, stack_id, target_stack_id, card_id)
    logging.debug("No direct move found")
    return []


def _can_be_moved_directly(board: Board, source_id, target_id, card_to_move):
    cloned_board = board.clone()
    available_dof = dof_board(cloned_board)
    source_stack = cloned_board.get_stack(source_id)
    target_stack = cloned_board.get_stack(target_id)

    if card_to_move < 0 or card_to_move >= len(source_stack.cards):
        logging.debug(f"_can_be_moved_directly: Invalid card index provided.")
        return False

    # TODO: I'm unsure this will fix all problems, probably it should be handled better
    if target_stack.is_empty() and available_dof > 0:
        available_dof -=1
    sequences = cards_to_sequences(source_stack.cards[card_to_move:])

    if not source_stack.is_stacked(card_to_move) or len(sequences) - 1 > available_dof:
        logging.debug(
            f"_can_be_moved_directly: Returning early as the input is invalid"
        )
        return False
    if not target_stack.can_stack(source_stack.cards[card_to_move]):
        return False

    return True


def _move_card_to_no_splits(
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
    # TODO: This is not the most optimal movement strategy
    moves: list[Move] = []
    cloned_board = board.clone()
    source_stack = cloned_board.get_stack(source_id)
    sequences = cards_to_sequences(source_stack.cards[card_to_move:])
    logging.debug(f"_move_card_to_no_intermediates: sequences = {sequences}")

    if not _can_be_moved_directly(board,source_id,target_id, card_to_move):
        return moves

    if len(sequences) > 1:
        stack_to_stack_moves = _optimal_stacked_reversible_movement(
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


def move_card_to_top(board: Board, source_id, target_id, card_id) -> list[Move]:
    """Produces a series of moves which lead to moving one specific card to the top of another stack through reversible moves"""
    func_name = "move_card_to_top"
    moves: list[Move] = []
    cloned_board = board.clone()

    source_stack = cloned_board.get_stack(source_id)
    target_stack = cloned_board.get_stack(target_id)

    logging.debug(f"{func_name}: Attempting to move card from stack {source_id} to stack {target_id}, card index: {card_id}")

    # Ensure the card to move is within the stack
    if card_id >= len(source_stack.cards):
        logging.error(f"{func_name}: Card index out of stack bounds")
        raise ValueError("Card index out of stack")
    if not target_stack.can_stack(source_stack.cards[card_id]):
        logging.debug(f"{func_name}: Target stack cannot accept the card, no moves made")
        return moves

    while True:
        if _can_be_moved_directly(cloned_board, source_id, target_id, card_id):
            direct_moves = _move_card_to_no_splits(
                cloned_board, source_id, target_id, card_id
            )
            moves.extend(direct_moves)
            logging.debug(f"{func_name}: Card moved directly with moves: {direct_moves}")
            break

        if card_id >= len(source_stack.cards) - 1:
            logging.debug(f"{func_name}: No more cards to move, exiting loop")
            moves = []
            break

        # Attempt reversible moves with and without considering splits
        reversible_moves = _reversible_move_away_from_stack(
            cloned_board,
            source_id,
            card_id+1,
            [source_id],
        )

        if reversible_moves:
            logging.debug(f"{func_name}: Reversible moves found: {reversible_moves}")
            moves.extend(reversible_moves)
            cloned_board.execute_moves(reversible_moves)
        else:
           logging.debug(f"{func_name}: No reversible moves found, attempting to free up space")
           # If no reversible moves are found, attempt to free up space
           freeing_moves = free_stack(
               cloned_board, ignored_stacks=[source_id, target_id]
           )
           if not freeing_moves:
               logging.debug(f"{func_name}: No moves to free up space, exiting")
               moves = []
               break
           logging.debug(f"{func_name}: Freeing moves found: {freeing_moves}")
           moves.extend(freeing_moves)
           cloned_board.execute_moves(freeing_moves)

    logging.debug(f"{func_name}: Total moves made: {moves}")
    return moves


def _optimal_stacked_reversible_movement(
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


def _select_stack_to_free(board: Board, ignored_stacks: list[int]) -> int|None:
    highest_rank = -1
    selected_stack_id = None
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
                get_top_cards_board(board, [id]),
            )

            if dof_needed <= available_dof:
                highest_rank = rank
                selected_stack_id = id
    return selected_stack_id


def find_placement_index_for_card(
    card: Card, stack: Stack, should_sequence=False
) -> int | None:
    """
    Finds the position within a given stack where a card can be legally placed according to the game's stacking rules.
    It considers only the visible and accessible cards.

    :param card: The card object to be placed.
    :param stack: The stack (list of card objects) in which to place the card.
    :return: The position (index) in the stack where the card can be placed according to the stacking rules.
             Returns None if there is no valid position for the card in this stack.
    """
    accessible_cards = stack.get_stacked()
    logging.debug(
        f"find_placement_index_for_card: accessible_cards = {accessible_cards}"
    )
    if not accessible_cards:
        logging.debug(f"find_placement_index_for_card: No accessible cards found")
        return None

    accessible_cards.reverse()

    for i, accessible_card in enumerate(accessible_cards):
        if not should_sequence and accessible_card.can_stack(card):
            return len(stack.cards) - i - 1
        elif should_sequence and accessible_card.can_sequence(card):
            return len(stack.cards) - i - 1

    return None

def find_partially_stackable(
    board: Board, sequence: list[Card], ignored_stacks: list[int]
) -> tuple[int, int]:
    """
    Finds a stack where a part of the given sequence can be stacked, starting from the second card in the sequence.

    Parameters:
    - board (Board): The current state of the board.
    - sequence (list[Card]): The sequence of cards to check for partial stackability.
    - ignored_stacks (list[int]): Indices of stacks to be ignored during the search.

    Returns:
    - tuple[int, int]: A tuple containing the index of the stack where the sequence can be partially stacked and the index within the sequence where stacking can start. Returns (-1, -1) if no such stack is found.
    """
    for sequence_index, card in enumerate(sequence, start=1): # start=1 to skip the top card
        target_stack = find_stack_to_move_sequence(board, card, ignored_stacks)
        if target_stack:
            return target_stack, sequence_index

    return (-1, -1)


def find_stacks_to_move_card(board: Board, card: Card, ignored_stacks=[], ignore_empty=False) -> list[int]:
    """Return a list of stack id on which a given card may be moved"""
    suitable_stacks = []
    for stack_id, stack in enumerate(board.stacks):
        if stack_id in ignored_stacks or (ignore_empty and stack.is_empty()):
            continue
        if stack.can_stack(card):
            suitable_stacks.append(stack_id)
    return suitable_stacks


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
