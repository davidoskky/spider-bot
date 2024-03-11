from __future__ import annotations

import logging
from collections import deque, namedtuple
from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence

from cardSequence import CardSequence, cards_to_sequences
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


def find_progressive_actions_manual(board: Board) -> list[list[Move]]:
    """
    Return a set of paths which lead to discovering a card which was previously not stacked.
    The last move is the one discovering the card.
    """
    paths: list[list[Move]] = []

    for source_stack_index, source_stack in enumerate(board.stacks):
        if source_stack.is_empty():
            continue
        first_stacked_card_id = source_stack.first_card_of_valid_stacked()

        # Ignore stacks resting on the board, nothing to be discovered there
        if first_stacked_card_id == 0:
            continue
        # print(f"find_progressive_actions_manual: source: {source_stack_index}, card {first_stacked_card_id}")

        card_to_move = board.get_card(source_stack_index, first_stacked_card_id)

        for target_stack_index, target_stack in enumerate(board.stacks):
            if source_stack_index == target_stack_index:
                continue

            target_card_index = find_placement_index_for_card(
                card_to_move, target_stack, should_sequence=False
            )

            if target_card_index is None:
                continue

            moves = move_card_splitting(
                board, source_stack_index, target_stack_index, first_stacked_card_id
            )
            if moves:
                paths.append(moves)

    # print(paths)
    return paths


def find_improved_equivalent_position(board: Board) -> list[Move]:
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
            if card.rank == 14:  # The king cannot be moved reversibly
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


def find_move_increasing_stacked_length_manual(board: Board) -> list[Move]:
    """
    Finds a set of reversible moves leading to a board position with an increased stacked length,
    without reducing the sequence length or the number of empty stacks.

    The objective of this function is to move cards towards stacks that start on a higher card.
    The rationale is that having long sequences starting on a king leaves other stacks more free
    of other movement options.
    This is not a move-wise cheap strategy but it should be safer.

    Parameters:
    - board (Board): The current state of the board.

    Returns:
    - List[Move]: A list of moves that result in an improved board position, or an empty list if no such move exists.
    """
    # TODO: Not strictly necessary, but this can be made more efficient, also move-wise by first sorting the target stacks by first accessible card
    # This should also favor moving to stacks where there is a king on the ground
    for target_stack_index, target_stack in enumerate(board.stacks):
        if target_stack.is_empty():
            continue

        target_stack_rank = target_stack.cards[
            target_stack.first_card_of_valid_stacked()
        ].rank

        for source_stack_index, source_stack in enumerate(board.stacks):
            if source_stack.is_empty() or target_stack_index == source_stack_index:
                continue

            logging.debug(
                f"find_move_increasing_stacked_length_manual: Evaluating source stack {source_stack_index}"
            )

            source_stack_rank = source_stack.cards[
                source_stack.first_card_of_valid_stacked()
            ].rank
            # Skip if moving towards a target stack which is lower ranking
            if source_stack_rank > target_stack_rank:
                logging.debug(
                    f"find_move_increasing_stacked_length_manual: Target stack {target_stack_index} rank lower than shource {source_stack_index}, breaking target {target_stack_rank}, source {source_stack_rank}"
                )
                continue

            first_card_to_consider = source_stack.first_card_of_valid_stacked()
            if first_card_to_consider > 0:
                # Don't consider the card if it is stacked over a covered one
                first_card_to_consider += 1

            logging.debug(
                f"find_move_increasing_stacked_length_manual: considering cards = {source_stack.cards[first_card_to_consider:]}"
            )

            for card_index in range(first_card_to_consider, len(source_stack.cards)):
                card = source_stack.cards[card_index]
                logging.debug(
                    f"find_move_increasing_stacked_length_manual: considering card {card}"
                )
                if card.rank == 14:  # The king cannot be moved reversibly
                    continue
                # Do not try moving cards which are in a sequence
                if source_stack.is_in_sequence(card_index):
                    logging.debug(
                        f"find_move_increasing_stacked_length_manual: skipping source card in sequence"
                    )
                    continue

                card_covering_target_index = find_placement_index_for_card(
                    card, target_stack, should_sequence=False
                )
                if card_covering_target_index is None:
                    continue

                target_card_index = card_covering_target_index - 1

                if card_covering_target_index < len(
                    target_stack.cards
                ) or target_stack.is_in_sequence(target_card_index):
                    logging.debug(
                        f"find_move_increasing_stacked_length_manual: skipping target card in sequence"
                    )
                    continue

                logging.debug(
                    f"find_move_increasing_stacked_length: target_card_ind = {target_card_index}"
                )

                if is_stacked_improved(
                    source_stack, target_stack, card_index, target_card_index
                ):
                    logging.debug(
                        f"find_move_increasing_stacked_length: source = {source_stack_index}, target = {target_stack_index}, card = {card_index}"
                    )
                    moves = move_cards_removing_interfering(
                        board, source_stack_index, target_stack_index, card_index
                    )
                    if moves:
                        return moves

    return []


def find_move_increasing_stacked_length(board: Board):
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

    logging.debug(
        f"is_sequence_improved: source card: {source_card} source: {source_sequence}, target = {target_sequence}"
    )

    target_rank_higher = target_sequence[0].rank > source_sequence[0].rank
    source_not_in_target = source_card.rank < target_sequence[-1].rank

    logging.debug(
        f"is_sequence_improved: target rank higher: {target_rank_higher}, source not in target = {source_not_in_target}"
    )

    return target_rank_higher and source_not_in_target


def is_stacked_improved(
    source_stack: Stack, target_stack: Stack, source_card_id: int, target_card_id: int
) -> bool:
    """
    Determines if moving a card or sequence of cards starting from a given index in the source stack to a target index in the target stack results in a valid and improved stacking by merging accessible sequences from both stacks.

    :param source_stack: The stack from which cards are moved.
    :param target_stack: The stack to which cards are moved.
    :param source_card_id: The starting index of the card(s) being moved in the source stack.
    :param target_card_id: The target position in the target stack for the bottom-most card being moved.
    :return: True if the resulting sequence in the target stack is valid and longer than the original, False otherwise.
    """

    if not source_stack.is_stacked(source_card_id) or not target_stack.is_stacked(
        target_card_id
    ):
        raise ValueError("Invalid card accessed")

    source_sequence = source_stack.cards[source_card_id:]
    target_sequence = target_stack.cards[
        target_stack.first_card_of_valid_stacked() : target_card_id + 1
    ]
    remaining_source = source_stack.cards[
        source_stack.first_card_of_valid_stacked() : source_card_id + 1
    ]
    leaving_target = target_stack.cards[target_card_id + 1 :]

    logging.debug(
        f"is_stacked_improved: source_seq = {source_sequence}, target_sequence = {target_sequence}"
    )

    if not source_sequence:
        logging.debug(f"is_stacked_improved: No source sequence")
        return False
    if not target_sequence:
        logging.debug(f"is_stacked_improved: No target sequence")
        return False
    if source_stack.get_stacked()[0].rank >= target_stack.get_stacked()[0].rank:
        logging.debug(
            f"is_stacked_improved: The source stack starts with a higher rank"
        )
        return False
    if target_stack.is_in_sequence(target_card_id):
        logging.debug(f"is_stacked_improved: The target card is in a sequence")
        return False

    if not target_sequence[-1].can_stack(source_sequence[0]):
        logging.debug(f"is_stacked_improved: Cannot stack")
        return False

    logging.debug(
        f"is_stacked_improved: len target_seq = {len(target_sequence)}, len remaining_source = {len(remaining_source)}"
    )

    return len(source_sequence) > len(leaving_target)


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
    # TODO: The logic should be simplified
    moves: list[Move] = []
    cloned_board = board.clone()

    target_stack = cloned_board.get_stack(target_stack_id)
    moving_card = cloned_board.get_card(source_stack_id, source_card_id)

    target_card_id = find_placement_index_for_card(moving_card, target_stack)
    logging.debug(f"move_cards_removing_interfering: target_card_id = {target_card_id}")
    # Check if the card can be placed in the target stack
    if target_card_id is None:
        return moves

    source_sequences = cloned_board.get_stack(source_stack_id).get_accessible_sequences(
        source_card_id
    )

    # Clear any cards covering the target position
    # Maybe this should be done by sequence as well
    clearance_moves = _move_stacked_to_temporary_position(
        cloned_board, target_stack_id, target_card_id
    )
    logging.debug(
        f"move_cards_removing_interfering: clearance_moves = {clearance_moves}"
    )
    cloned_board.execute_moves(clearance_moves)

    moves_to_complete_switch = move_card_splitting(
        cloned_board, source_stack_id, target_stack_id, source_card_id
    )
    if moves_to_complete_switch:
        moves.extend(clearance_moves)
        moves.extend(moves_to_complete_switch)
    elif len(source_sequences) > 1:
        cloned_board = board.clone()
        covering_sequence_id = source_sequences[1].start_index
        clearance_moves = _move_stacked_to_temporary_position(
            cloned_board,
            source_stack_id,
            covering_sequence_id,
            ignored_stacks=[target_stack_id],
        )
        cloned_board.execute_moves(clearance_moves)
        second_clearance_moves = _move_stacked_to_temporary_position(
            cloned_board,
            target_stack_id,
            target_card_id,
            ignored_stacks=[source_stack_id],
        )
        cloned_board.execute_moves(second_clearance_moves)
        moves_to_complete_switch = move_card_splitting(
            cloned_board, source_stack_id, target_stack_id, source_card_id
        )
        if moves_to_complete_switch:
            moves.extend(clearance_moves)
            moves.extend(second_clearance_moves)
            moves.extend(moves_to_complete_switch)
        else:
            first_sequence = source_sequences[0]
            for card_index in range(
                source_card_id + 1, source_card_id + len(first_sequence)
            ):
                cloned_board = board.clone()
                individual_clearance_moves = _move_stacked_to_temporary_position(
                    cloned_board,
                    source_stack_id,
                    card_index,
                    ignored_stacks=[target_stack_id],
                )
                cloned_board.execute_moves(individual_clearance_moves)
                second_clearance_moves = _move_stacked_to_temporary_position(
                    cloned_board,
                    target_stack_id,
                    target_card_id,
                    ignored_stacks=[source_stack_id],
                )
                cloned_board.execute_moves(second_clearance_moves)

                # Try the main move again after each individual card move
                individual_moves_to_complete_switch = move_card_splitting(
                    cloned_board, source_stack_id, target_stack_id, source_card_id
                )
                if individual_moves_to_complete_switch:
                    moves.extend(individual_clearance_moves)
                    moves.extend(second_clearance_moves)
                    moves.extend(individual_moves_to_complete_switch)
                    break

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


def get_top_cards_board(board: Board, ignored_stacks: list[int] = []) -> list[Card]:
    """
    Retrieves the top card from each stack in the board, excluding specified stacks and empty ones.

    Parameters:
    - board (Board): The current state of the board.
    - ignored_stacks (List[int]): Indices of stacks to be ignored.

    Returns:
    - List[Card]: A list of top cards from the stacks not ignored and not empty.
    """
    filtered_stacks = [
        stack for i, stack in enumerate(board.stacks) if i not in ignored_stacks
    ]

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


def can_move_stacked_reversibly(
    stacked_cards: list[CardSequence], stacks: list[Stack], empty_stacks=0, strict=False
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
    first_stacked: list[CardSequence],
    second_stacked: list[CardSequence],
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
    sequences: list[CardSequence], top_cards: list[Card]
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

    for sequence in reversed(sequences):
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
    logging.debug(
        f"_move_stacked_to_temporary_position: attempting to move card {card_index} from stack {stack_id}"
    )
    moves = []
    clearing_moves = []
    cloned_board = board.clone()
    stack = cloned_board.get_stack(stack_id)
    card = cloned_board.get_card(stack_id, card_index)
    if card is None:
        logging.debug(f"_move_stacked_to_temporary_position: card index out of bounds")
        return []

    temporary_stack_ids = find_stacks_to_move_card(
        cloned_board,
        stack.cards[card_index],
        ignored_stacks=ignored_stacks,
        ignore_empty=True,
    )
    if not temporary_stack_ids:
        temporary_stack_ids = find_stacks_to_move_card(
            cloned_board,
            stack.cards[card_index],
            ignored_stacks=ignored_stacks,
            ignore_empty=False,
        )

        if not temporary_stack_ids:
            clearing_moves = free_stack(cloned_board, ignored_stacks=ignored_stacks)
            if not clearing_moves:
                return []

            cloned_board.execute_moves(clearing_moves)
            if not cloned_board.card_exists(stack_id, card_index):
                stack_id, card_index = cloned_board.card_position(card)
            temporary_stack_ids = find_stacks_to_move_card(
                cloned_board, card, ignored_stacks=ignored_stacks, ignore_empty=False
            )

    logging.debug(
        f"_move_stacked_to_temporary_position: plausible stacks = {temporary_stack_ids}"
    )

    for temporary_stack_id in temporary_stack_ids:
        if temporary_stack_id in ignored_stacks:
            continue
        logging.debug(
            f"_move_stacked_to_temporary_position: stack = {stack_id}, card = {card_index}, temporary = {temporary_stack_id}"
        )
        moves = move_card_splitting(
            cloned_board, stack_id, temporary_stack_id, card_index
        )
        if moves:
            break

    if clearing_moves and moves:
        clearing_moves.extend(moves)
        moves = clearing_moves

    logging.debug(f"_move_stacked_to_temporary_position: moves = {moves}")

    return moves


def free_stack(
    board: Board, ignored_stacks: list[int] = [], stacks_not_to_move_to=[]
) -> list[Move]:
    moves: list[Move] = []
    cloned_board = board.clone()
    initial_empty_stacks = cloned_board.count_empty_stacks()

    stacks_to_free_ids = _select_stacks_to_free(cloned_board)

    if not stacks_to_free_ids:
        logging.debug(f"free_stack: No stack can be freed")
        return moves

    for stack_id in stacks_to_free_ids:
        if stack_id in ignored_stacks:
            continue

        logging.debug(f"Trying to free stack = {stack_id}")

        target_stack_id = find_stack_to_move_sequence(
            cloned_board, board.get_card(stack_id, 0), ignored_stacks, ignore_empty=True
        )

        if target_stack_id is None or target_stack_id in stacks_not_to_move_to:
            logging.debug("free_stack: No target stack found, cannot move the sequence")
            continue

        temp_moves = move_card_splitting(board, stack_id, target_stack_id, 0)

        logging.debug(f"free_stack: temp_moves = {temp_moves}")

        if not temp_moves:
            continue

        cloned_board.execute_moves(temp_moves)
        plausible_moves = []
        plausible_moves.extend(temp_moves)

        while cloned_board.count_empty_stacks() <= initial_empty_stacks:
            more_moves = free_stack(cloned_board, ignored_stacks, stacks_not_to_move_to)
            if not more_moves:
                plausible_moves = []
                cloned_board = board.clone()
                break
            plausible_moves.extend(more_moves)
            cloned_board.execute_moves(more_moves)

        if plausible_moves:
            moves = plausible_moves
            break

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


def move_card_splitting(
    board: Board, source_id: int, target_id: int, card_to_move: int
) -> list[Move]:
    """Move a card to another stack moving sequences on top of it on free stacks or on top of other cards"""
    cloned_board = board.clone()
    moves = []

    card = cloned_board.get_card(source_id, card_to_move)
    if card is None:
        logging.error(
            f"move_card_splitting: Card index {card_to_move} out of stack {source_id} bounds"
        )
        cloned_board.display_game_state()
        raise ValueError("Card index out of stack")

    if not cloned_board.get_stack(target_id).can_stack(card):
        logging.debug(
            f"move_card_splitting: Target stack cannot accept the card, no moves made"
        )
        return moves

    sequences = cloned_board.get_stack(source_id).get_accessible_sequences(
        first_card_id=card_to_move
    )
    logging.debug(f"Sequences: {sequences}")
    if not sequences:
        return []

    available_dof = dof_board(cloned_board)
    last_available_dof = available_dof
    if cloned_board.get_stack(target_id).is_empty():
        last_available_dof = max(0, last_available_dof - 1)

    # We know the first sequence is movable as we have previously checked
    movable_sequences = [0]

    # Reverse the iteration of sequences, starting from the last down to the second sequence
    # Note: The first sequence is skipped as it's handled separately
    # The flag 'cards_above' indicates if there are cards above the current sequence that block movement
    cards_above = False
    for i in range(len(sequences) - 1, 0, -1):
        sequence = sequences[i]

        # When there are no cards above, consider only the first card of the sequence, useless to move the other ones
        if not cards_above:
            card = sequence[0]
            available_stacks = find_stacks_to_move_card(
                cloned_board, card, ignore_empty=True
            )
            if available_stacks:
                movable_sequences.append(i)
                continue
            else:
                cards_above = True
        else:
            for j, card in enumerate(sequence):
                available_stacks = find_stacks_to_move_card(
                    cloned_board, card, ignore_empty=True
                )
                if available_stacks:
                    movable_sequences.append(i)
                    # Update the blocking flag, False if we moved the first card
                    cards_above = j != 0
                    break

    movable_sequences.sort()

    differences: list[int] = []
    for i, j in zip(movable_sequences[1:], movable_sequences):
        differences.append(i - j - 1)
    if movable_sequences[-1] != len(sequences) - 1:
        differences.append(len(sequences) - 1 - movable_sequences[-1])

    possible = False

    if last_available_dof >= len(sequences) - 1:
        possible = True

    if not possible and not differences:
        return []

    if not possible and differences[0] > last_available_dof:
        return []
    if not possible and max(differences) > available_dof:
        return []

    # This while loop is dangerous as it could get us into an infinite loop.
    # The condition for success should be checked very well before it.
    # Take extreme care when editing this function.
    while sequences:
        if last_available_dof >= len(sequences) - 1:
            moves.extend(
                _move_card_to_no_splits(
                    cloned_board, source_id, target_id, card_to_move
                )
            )
            break

        previous_iteration_length = len(sequences)
        # Iteratively remove full sequences until possible. Else, attempt moving single cards.
        # From the first accessible sequence attempt to go towards the topmost one, unless
        # there are more in a row than available DoF which cannot be moved.
        move_made = False
        for i in range(len(sequences) - 1, len(sequences) - 1 - available_dof - 1, -1):
            sequence = sequences[i]
            logging.debug(f"Sequence considered in iteration: {sequence}")
            available_stacks = find_stacks_to_move_card(
                cloned_board, sequence.top_card(), ignore_empty=True
            )

            if available_stacks:
                partial_moves = _move_card_to_no_splits(
                    cloned_board,
                    source_id,
                    available_stacks[0],
                    sequence.start_index,
                )
                moves.extend(partial_moves)
                cloned_board.execute_moves(partial_moves)
                del sequences[i:]
                move_made = True
                break

        # If there are still sequences, then we have to move things one card at a time.
        # Attempt to move the topmost card which can be moved.
        # The topmost accessible sequence should not be considered, but that is not necessary as it
        # Should be handled above, as such, if we observe it it may be good to raise an Error
        # At this point, sequences should be longer that the available DoF, otherwise an error appened somewhere
        if not move_made:
            if last_available_dof >= len(sequences) - 1:
                raise SystemError("This should never happen. Please, fix my algorithm.")

            # Starting from the topmost attempt to move single cards in the most bottom
            # sequences, considering a number of sequences equal to the DoF
            # Don't consider the last sequence as it makes no difference to move a card from there
            for i in range(len(sequences) - available_dof - 1, len(sequences) - 1):
                sequence = sequences[i]
                for j, card in enumerate(sequence, start=0):
                    available_stacks = find_stacks_to_move_card(
                        cloned_board, card, ignore_empty=True
                    )

                    if available_stacks:
                        partial_moves = _move_card_to_no_splits(
                            cloned_board,
                            source_id,
                            available_stacks[0],
                            sequence.start_index + j,
                        )
                        moves.extend(partial_moves)
                        cloned_board.execute_moves(partial_moves)
                        # Delete sequences following the current one. This one has not been fully moved yet.
                        # Deleting the cards from the sequence is not required as it won't be considered on the
                        # Following iteration.
                        del sequences[i + 1 :]
                        move_made = True
                        break
                if move_made:
                    break

        if previous_iteration_length == len(sequences):
            raise SystemError("This should never happen. Please, fix my algorithm.")

    return moves


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
        available_dof -= 1
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
    target_is_empty = cloned_board.get_stack(target_id).is_empty()
    sequences = cards_to_sequences(source_stack.cards[card_to_move:])
    logging.debug(f"_move_card_to_no_splits: sequences = {sequences}")

    logging.debug(
        f"source: {source_id}, target: {target_id}, card: {card_to_move}, sequences: {sequences}"
    )

    if not _can_be_moved_directly(board, source_id, target_id, card_to_move):
        return moves

    if len(sequences) > 1 and not target_is_empty:
        stack_to_stack_moves = _optimal_stacked_reversible_movement(
            cloned_board, source_id, len(sequences) - 1
        )
        if not stack_to_stack_moves:
            return []

        for move_set in stack_to_stack_moves:
            start, dest = move_set
            card_id = cloned_board.stacks[start].first_card_of_valid_sequence()
            if start == source_id and card_id < card_to_move:
                card_id = card_to_move
            move = Move(start, dest, card_id)
            cloned_board.move_by_index(*move)
            moves.append(move)
    if len(sequences) > 1 and target_is_empty:
        stack_to_stack_moves = _optimal_stacked_reversible_movement(
            cloned_board, source_id, len(sequences), final_stack=target_id
        )
        if not stack_to_stack_moves:
            return []

        for move_set in stack_to_stack_moves:
            start, dest = move_set
            card_id = cloned_board.stacks[start].first_card_of_valid_sequence()
            if start == source_id and card_id < card_to_move:
                card_id = card_to_move
            move = Move(start, dest, card_id)
            cloned_board.move_by_index(*move)
            moves.append(move)

    if len(sequences) == 1 and target_is_empty:
        move = Move(source_id, target_id, card_to_move)
        moves.append(move)

    if not target_is_empty:
        move = Move(source_id, target_id, card_to_move)
        moves.append(move)
    logging.debug(f"_move_card_to_no_splits: moves = {moves}")
    return moves


def _optimal_stacked_reversible_movement(
    board: Board,
    source_stack_id: int,
    amount_of_sequences: int,
    final_stack: int | None = None,
) -> list[tuple[int, int]]:
    """
    Generates an optimal sequence of moves to free one stack given a number of empty stacks,
    ensuring that the number of moves starting from the source stack matches the amount of sequences.
    It does not consider moving cads on top of others, it only uses the empty stacks.

    :param board: The current state of the Spider Solitaire game.
    :param source_stack_id: ID of the source stack from which cards are to be moved.
    :param amount_of_sequences: Number of sequences in the source stack to be moved.
    :param final_stack: Ensure that the last move ends on this stack or it is not used in any move
    :return: List of tuples representing the moves, where each tuple is (source_stack_id, target_stack_id).
    """
    empty_stacks = [id for id, stack in enumerate(board.stacks) if stack.is_empty()]
    if not empty_stacks:
        return []

    if amount_of_sequences > 2 ^ (len(empty_stacks)) - 1:
        return []

    if amount_of_sequences < 1:
        return []

    stack_to_stack_moves, _ = destacker(
        amount_of_sequences, empty_stacks, source_stack_id
    )
    logging.debug(
        f"_optimal_stacked_reversible_movement: stack moves = {stack_to_stack_moves}"
    )
    if final_stack is not None:
        stack_to_stack_moves = _translate_optimal_moves(
            stack_to_stack_moves, final_stack
        )
        logging.debug(
            f"_optimal_stacked_reversible_movement: stack moves after translation = {stack_to_stack_moves}"
        )
    return stack_to_stack_moves


def destacker(sequences: int, empty_stacks: list[int], source_id: int):
    """Algorithm 1: Move k sequences into n empty stacks

    Algorithm described somewhere in the reference documentation.
    """

    if sequences == 0:
        return [], []

    if not empty_stacks:
        raise ValueError("No empty stacks provided")

    used_stacks = []
    moves = []
    while empty_stacks:
        # Using the maximum ensures always having at least 1, which covers the case in which only one stack is empty, in which one card has to be moved
        to_move = max(2 ^ (len(empty_stacks) - 2), 1)
        stacks_to_free = []

        for _ in range(to_move):
            if sequences == 0:
                break
            target_stack = empty_stacks.pop()
            used_stacks.append(target_stack)
            stacks_to_free.append(target_stack)
            moves.append((source_id, target_stack))
            sequences -= 1

            logging.debug(f"moves: {moves}")

        if sequences == 0:
            return moves, used_stacks

        # Free up stacks for the next iteration by moving all sequence to a single stack
        destination_stack = stacks_to_free.pop()
        logging.debug(f"to free {stacks_to_free}")
        for origin in reversed(stacks_to_free):
            moves.append((origin, destination_stack))
            empty_stacks.append(origin)
            used_stacks.remove(origin)

    # At this point, all stacks are occupied and we have 2^(n-1) sequences on the stacks being considered
    # Now we perform the moves to move all those sequences to one single stack.
    logging.debug(f"Moves when destacking complete: {moves}")
    cycle = 1
    step = 1

    # Once the last stack has been occupied, stop considering it among the movable ones
    last_stack = used_stacks.pop()
    stacks_to_free = [used_stacks[-1]]
    while used_stacks:
        # At a maximum, two stacks can be stacked per destack cycle when all DoF are being used. First and last move only move two stacks.

        for origin in stacks_to_free:
            moves.append((origin, last_stack))
            empty_stacks.append(origin)
            used_stacks.remove(origin)

        logging.debug(f"moves after stacker {moves}, occupied_stacks: {used_stacks}")

        if not used_stacks:
            break

        if not empty_stacks:
            continue

        stacks_to_free.append(used_stacks[-1])
        t_moves, t_used = destacker(2 ^ (step) - 1, empty_stacks, last_stack)
        moves.extend(t_moves)
        used_stacks.extend(t_used)
        stacks_to_free.append(t_used[-1])

        step -= 1
        if step == 0:
            cycle += 1
            step = cycle

    t_moves, _ = destacker(sequences, empty_stacks, source_id)
    moves.extend(t_moves)
    logging.debug(f"moves 2: {moves}")
    return moves, used_stacks


def _translate_optimal_moves(
    moves: list[tuple[int, int]], target_id: int
) -> list[tuple[int, int]]:
    """
    Translates a sequence of moves so that the last move becomes goes to the desired target_id
    and intermediate stack IDs are shuffled accordingly.

    :param moves: Initial sequence of moves as a list of tuples (source_stack, target_stack).
    :param source_id: ID of the source stack.
    :param target_id: ID of the target stack.
    :return: Translated sequence of moves.
    """
    if not moves:
        return moves

    # If no move is going to the target id, no need to translate
    if target_id not in [move[-1] for move in moves]:
        return moves

    translated_moves = []
    original_final_stack = moves[-1][1]

    # Translate each move in the sequence
    for move in moves:
        src, dst = move
        if dst == original_final_stack:
            new_dst = target_id
        elif dst == target_id:
            new_dst = original_final_stack
        else:
            new_dst = dst

        if src == original_final_stack:
            new_src = target_id
        elif src == target_id:
            new_src = original_final_stack
        else:
            new_src = src

        translated_moves.append((new_src, new_dst))

    return translated_moves


def _select_stacks_to_free(board: Board) -> list[int]:
    stacks_to_free = []
    available_dof = dof_board(board)

    for id, stack in enumerate(board.stacks):
        if (
            stack.is_empty()
            or not stack.is_stacked_on_table()
            or stack.cards[0].rank == 14
        ):
            continue

        cards_to_move = stack.cards
        if not cards_to_move:
            continue

        stacks_to_consider = [
            cid
            for cid, other_stack in enumerate(board.stacks)
            if cid != id and not other_stack.is_empty()
        ]
        for dest in stacks_to_consider:
            if not board.stacks[dest].can_stack(cards_to_move[0]):
                continue

            dof_needed, _ = dof_to_move_stacked_reversibly(
                cards_to_sequences(cards_to_move),
                get_top_cards_board(board),
            )

            if dof_needed <= available_dof:
                stacks_to_free.append((id, stack.cards[0].rank))

    # Sort the list of stacks by rank in descending order (highest rank first)
    stacks_to_free.sort(key=lambda x: x[1], reverse=True)
    return [stack_id for stack_id, _ in stacks_to_free]


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
    if stack.is_empty():
        logging.debug("find_placement_index_for_card: Stack is empty.")
        return 0

    # first_accessible_card_id = stack.first_card_of_valid_stacked()
    # for i in range(len(stack.cards) -1, first_accessible_card_id - 1, -1):
    #     accessible_card = stack.cards[i]
    #
    #     if not should_sequence and accessible_card.can_stack(card):
    #         return i
    #     elif should_sequence and accessible_card.can_sequence(card):
    #         return i
    # return None
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
            return len(stack.cards) - i
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
    for sequence_index, card in enumerate(
        sequence, start=1
    ):  # start=1 to skip the top card
        target_stack = find_stack_to_move_sequence(board, card, ignored_stacks)
        if target_stack:
            return target_stack, sequence_index

    return (-1, -1)


def find_stacks_to_move_card(
    board: Board, card: Card, ignored_stacks=[], ignore_empty=False
) -> list[int]:
    """Return a list of stack id on which a given card may be moved

    The list is in order, the stacks which can sequence the card are listed first
    """
    sequencing_stacks = []
    stacking_stacks = []
    for stack_id, stack in enumerate(board.stacks):
        if stack_id in ignored_stacks or (ignore_empty and stack.is_empty()):
            continue
        if stack.can_sequence(card):
            sequencing_stacks.append(stack_id)
        if stack.can_stack(card):
            stacking_stacks.append(stack_id)

    suitable_stacks = sequencing_stacks
    suitable_stacks.extend(stacking_stacks)
    return suitable_stacks


def find_stack_to_move_sequence(
    board: Board, top_card: Card, ignored_stacks: list[int] = [], ignore_empty=False
) -> Optional[int]:
    """
    Find a stack to which the sequence can be legally moved.
    If ignore_empty is false, empty stacks will be considered, but if a valid non empty stacks is
    available it will be returned.

    :param board: The current board state.
    :param sequence: The sequence of cards to move.
    :return: ID of the stack where the sequence can be moved, or None if not found.
    """
    suitable_empty_stack = None

    for target_stack_id, target_stack in enumerate(board.stacks):
        if target_stack_id in ignored_stacks:
            continue

        if not target_stack.is_empty() and target_stack.can_stack(top_card):
            return target_stack_id

        if suitable_empty_stack is None and target_stack.is_empty():
            suitable_empty_stack = target_stack_id

    if not ignore_empty and suitable_empty_stack is not None:
        return suitable_empty_stack

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
