import copy
from operator import is_
from typing import Optional

from deck import Card, Deck, SimpleDeck
from moves_exploration import Move, find_progressive_actions


class Stack:
    """
    Represents a stack of cards in Spider Solitaire.
    """

    def __init__(self, cards: list[Card]):
        self.cards = cards
        self.first_visible_card = len(cards) - 1
        self.first_accessible_sequence = self.first_card_of_valid_sequence()

    def __repr__(self):
        representation = [
            repr(card) + " " if i >= self.first_visible_card else "XX "
            for i, card in enumerate(self.cards)
        ]
        return "".join(representation)

    def clone(self):
        cloned_stack = Stack(copy.copy(self.cards))
        cloned_stack.first_visible_card = self.first_visible_card
        cloned_stack.first_accessible_sequence = self.first_accessible_sequence
        return cloned_stack

    def is_visible(self, card_index: int) -> bool:
        """Check if the card at card_index is face-up (visible)."""
        return card_index >= self.first_visible_card

    def can_stack(self, card: Card) -> bool:
        return self.is_empty() or self.top_card().can_stack(card)

    def reveal_top_card(self):
        """Flip the top card to face-up."""
        if self.cards and self.first_visible_card >= len(self.cards):
            self.first_visible_card = len(self.cards) - 1

    def is_valid_suit_sequence(self, start_index: int) -> bool:
        return self._is_valid_sequence(
            start_index, lambda c, n: c.can_stack(n) and c.same_suit(n)
        )

    def is_valid_stacked_sequence(self, start_index: int) -> bool:
        return self._is_valid_sequence(start_index, lambda c, n: c.can_stack(n))

    def _is_valid_sequence(self, start_index: int, condition) -> bool:
        if not self.is_visible(start_index):
            return False
        return all(
            condition(card, next_card)
            for card, next_card in zip(
                self.cards[start_index:], self.cards[start_index + 1 :]
            )
        )

    def count_sequences_to_index(self, index: int) -> int:
        """
        Count the number of stacked sequences needed to reach a given index in a stack.

        :param stack: The stack of cards.
        :param index: The index of the card in the stack.
        :return: The number of stacked sequences to reach the card at the given index.
        """
        if index < 0 or index >= len(self.cards):
            raise ValueError("Index is out of range for the stack")

        if self.is_empty():
            return 0

        all_stacked_sequences = self.get_all_stacked_sequences()
        count = 0
        current_index = len(self.cards) - 1

        for sequence in reversed(all_stacked_sequences):
            sequence_length = len(sequence)
            if current_index - sequence_length < index:
                count += 1
                break
            count += 1
            current_index -= sequence_length

        return count

    def visible_cards(self) -> list[Card]:
        return self.cards[self.first_visible_card :]

    def hidden_cards(self):
        return self.cards[: self.first_visible_card]

    def is_movable(self, card_index: int) -> bool:
        return card_index >= self.first_accessible_sequence

    def add_sequence(self, sequence):
        """Add a sequence of cards to the stack"""
        self.cards.extend(sequence)
        self.first_accessible_sequence = self.first_card_of_valid_sequence()

    def pop_sequence(self, start_index):
        if not self.is_visible(start_index):
            raise ValueError("Cannot remove a hidden sequence")
        sequence = self.cards[start_index:]
        self.cards = self.cards[:start_index]
        self.reveal_top_card()
        self.first_accessible_sequence = self.first_card_of_valid_sequence()
        return sequence

    def top_card(self):
        return self.cards[-1] if self.cards else None

    def is_empty(self):
        return not self.cards

    def first_card_of_valid_sequence(self) -> int:
        """Find the index of the first card of the valid sequence from the top."""
        for i in range(self.first_visible_card, len(self.cards)):
            if self.is_valid_suit_sequence(i):
                return i
        return 0

    def valid_sequence_length(self) -> int:
        return len(self.cards) - self.first_card_of_valid_sequence()

    def first_card_not_in_sequence(self):
        """Find the index of the first card that is not in sequence from the top."""
        i = self.first_accessible_sequence
        if i > 0:
            return i - 1
        else:
            return None

    def get_sequence(self):
        return self.cards[self.first_accessible_sequence :]

    def first_card_of_valid_stacked(self) -> int:
        """Find the index of the first card of the valid sequence from the top."""
        for i in range(len(self.cards)):
            if self.is_valid_stacked_sequence(i):
                return i
        return 0

    def is_stacked(self, card_index: int) -> bool:
        """Returns if the given card index is in an accessible stacked sequence"""
        return card_index >= self.first_card_of_valid_stacked()

    def is_stacked_on_table(self):
        """Returns if the first card of stacked sequence is on top of empty board"""
        return self.first_card_of_valid_stacked() == 0

    def valid_stacked_length(self) -> int:
        return len(self.cards) - self.first_card_of_valid_stacked()

    def get_stacked(self) -> list[Card]:
        "Gets all the cards in the first accessible stacked sequence"
        return self.cards[self.first_card_of_valid_stacked() :]

    def get_all_stacked_sequences(self) -> list[list[Card]]:
        """
        Generate a list of lists, where each inner list contains the cards in a valid stacked sequence
        from each stack on the board. Ordered from the topmost sequence to the bottom.

        This function iterates through each stack on the board, examining the stacked cards.
        It creates a list for each valid stacked sequence and adds these lists to an outer list.

        :return: A list of lists, each containing the cards in a valid stacked sequence.
        """
        all_stacked_sequences: list[list[Card]] = []
        if self.is_empty():
            return all_stacked_sequences

        cards = self.visible_cards()
        current_sequence = [cards[0]]
        for i in range(1, len(cards)):
            if current_sequence[-1].can_stack(cards[i]):
                current_sequence.append(cards[i])
            else:
                all_stacked_sequences.append(current_sequence)
                current_sequence = [cards[i]]

        all_stacked_sequences.append(current_sequence)
        return all_stacked_sequences

    def get_visible_sequences(self):
        """
        Generate a list of lists, where each inner list contains the cards in a valid stacked sequence
        from each stack on the board.

        This function iterates through each stack on the board, examining the stacked cards.
        It creates a list for each valid stacked sequence and adds these lists to an outer list.

        :return: A list of lists, each containing the cards in a valid stacked sequence.
        """
        all_stacked_sequences = []
        if self.is_empty():
            return all_stacked_sequences

        cards = self.visible_cards()
        current_sequence = [cards[0]]
        for i in range(1, len(cards)):
            if cards[i - 1].can_sequence(cards[i]):
                current_sequence.append(cards[i])
            else:
                all_stacked_sequences.append(current_sequence)
                current_sequence = [cards[i]]

        all_stacked_sequences.append(current_sequence)
        return all_stacked_sequences

    def get_accessible_sequences(self) -> list[list[Card]]:
        """
        Generate a list of lists, where each inner list contains the cards in a valid sequences
        from each stack on the board.

        This function iterates through each stack on the board, examining the stacked cards.
        It creates a list for each valid stacked sequence and adds these lists to an outer list.

        :return: A list of lists, each containing the cards in a valid stacked sequence.
        """
        all_stacked_sequences = []

        cards = self.get_stacked()
        if cards:
            current_sequence = [cards[0]]
            for i in range(1, len(cards)):
                if cards[i - 1].can_sequence(cards[i]):
                    current_sequence.append(cards[i])
                else:
                    all_stacked_sequences.append(current_sequence)
                    current_sequence = [cards[i]]

            all_stacked_sequences.append(current_sequence)
        return all_stacked_sequences


class Board:
    INITIAL_STACKS_COUNT = 10
    CARDS_IN_SMALL_STACK = 5
    CARDS_IN_LARGE_STACK = 6

    def __init__(
        self,
        seed: Optional[int] = None,
        stacks: Optional[tuple[Stack, ...]] = None,
        deck: Optional[Deck] = None,
        completed_stacks: int = 0,
    ) -> None:
        if stacks is not None and deck is not None:
            self._initialize_for_cloning(stacks, deck, completed_stacks)
        else:
            self._initialize_new_game(seed)

    def _initialize_for_cloning(
        self, stacks: tuple[Stack, ...], deck, completed_stacks
    ):
        self.deck = deck.clone()
        self.stacks = tuple([stack.clone() for stack in stacks])
        self.completed_stacks = [1] * completed_stacks

    def _initialize_new_game(self, seed):
        self.deck = Deck(seed)
        self.stacks: tuple[Stack, ...]
        self.completed_stacks = []

        generated_stacks = []
        for i in range(Board.INITIAL_STACKS_COUNT):
            num_face_down = (
                Board.CARDS_IN_LARGE_STACK if i < 4 else Board.CARDS_IN_SMALL_STACK
            )
            stack_cards = self.deck.draw(num_face_down)
            stack = Stack(stack_cards)

            generated_stacks.append(stack)

        self.stacks = tuple(generated_stacks)

    def clone(self):
        """Create a new Board object with a deep clone of the card disposition."""
        return Board(stacks=self.stacks, deck=self.deck.clone())

    def is_valid_move(
        self, from_stack: Stack, to_stack: Stack, card_index: int
    ) -> bool:
        if from_stack.is_empty():
            return False

        if card_index == 0 and to_stack.is_empty():
            return False
        return to_stack.can_stack(
            from_stack.cards[card_index]
        ) and from_stack.is_valid_suit_sequence(card_index)

    def move(self, source_stack: Stack, destination_stack: Stack, card_index: int):
        """Move a sequence of cards from one stack to another"""
        if not self.is_valid_move(source_stack, destination_stack, card_index):
            source_id = self.stacks.index(source_stack)
            destination_id = self.stacks.index(destination_stack)
            self.display_game_state()
            raise ValueError(
                f"Invalid move attempted. From {source_id}, To {destination_id}, card {card_index}"
            )
        sequence_to_move = source_stack.pop_sequence(card_index)
        destination_stack.add_sequence(sequence_to_move)
        self.check_for_completion(destination_stack)

    def move_by_index(self, source_stack_id, destination_stack_id, card_index):
        """Move a sequence of cards from one stack to another"""
        self.move(
            self.stacks[source_stack_id], self.stacks[destination_stack_id], card_index
        )

    def execute_moves(self, moves: list[Move]):
        for move in moves:
            self.move_by_index(*move)

    def is_game_won(self) -> bool:
        """Check if the game has been won"""
        return not self.deck.cards and not any(stack.cards for stack in self.stacks)

    def is_game_lost(self):
        """Check if no more moves are available"""
        if self.deck.cards:
            return False

        actions = find_progressive_actions(self)
        if actions:
            return False

        return True

    def draw_from_deck(self):
        """Draw more cards from the deck"""
        if self.deck.cards:
            for stack in self.stacks:
                drawn_card = self.deck.draw(1)
                stack.add_sequence(drawn_card)
                self.check_for_completion(stack)

            return True
        else:
            return False

    def check_for_completion(self, stack: Stack):
        """Check and update for any completed stacks in the given stack"""
        self.just_completed_stack = False
        if len(stack.cards) >= 13 and stack.is_valid_suit_sequence(
            len(stack.cards) - 13
        ):
            completed_sequence = stack.pop_sequence(len(stack.cards) - 13)
            self.completed_stacks.append(completed_sequence)
            self.just_completed_stack = True

    def list_available_moves(self):
        """List all available moves in the current game state."""
        available_moves = self._list_stack_to_stack_moves() + self._list_deck_moves()
        return available_moves

    def _list_stack_to_stack_moves(self):
        """List all valid stack-to-stack moves."""
        moves = []
        for i, from_stack in enumerate(self.stacks):
            for card_index in range(
                from_stack.first_accessible_sequence, len(from_stack.cards)
            ):
                moves.extend(self._get_valid_moves_to_other_stacks(i, card_index))
        return moves

    def _get_valid_moves_to_other_stacks(self, from_index, card_index):
        """Get all valid moves of one cart in one stack to other stacks. For optimization reasons only one empty stack is listed"""
        used_empty_stack = False
        valid_moves = []
        for to_index, to_stack in enumerate(self.stacks):
            move = Move(from_index, to_index, card_index)
            if used_empty_stack and to_stack.is_empty():
                continue
            if to_stack.is_empty():
                used_empty_stack = True
            if self.is_valid_move(self.stacks[from_index], to_stack, card_index):
                valid_moves.append(move)
        return valid_moves

    def get_empty_stack_id(self):
        for id, stack in enumerate(self.stacks):
            if stack.is_empty():
                return id
        return -1

    def _list_deck_moves(self):
        """List moves related to the deck, if any."""
        return [("draw_from_deck",)] if self.deck.cards else []

    def list_indifferent_moves(self) -> list[Move]:
        available_moves = self._list_stack_to_stack_moves()
        indifferent_moves = []
        for move in available_moves:
            if self.is_move_indifferent(move):
                indifferent_moves.append(move)
        return indifferent_moves

    def is_move_indifferent(self, move: Move) -> bool:
        """
        Determine if a move is indifferent regarding the future freedom of moves.

        A move is considered indifferent in two cases:
        1. Moving the topmost visible card from one stack to another, regardless of the card it is moved to.
        2. Moving any card from a stack where the immediately lower card (if visible) can stack on the moved card.

        :param move: The move to be evaluated, represented as a Move object.
        :return: True if the move is indifferent, False otherwise.
        """
        if move.card_index == 0:
            return True
        source_stack = self.stacks[move.source_stack]

        if source_stack.is_visible(move.card_index - 1) and source_stack.cards[
            move.card_index - 1
        ].can_stack(source_stack.cards[move.card_index]):
            return True

        return False

    def display_game_state(self):
        """Display the current state of the game"""
        for i, stack in enumerate(self.stacks):
            print(f"Stack {i}: ", end="")
            print(stack)
        print(
            f"Missing Deals: {len(self.deck.cards)/10}Completed Stacks: {len(self.completed_stacks)}"
        )

    def get_state(self):
        """Return the current game state as a list of lists (each list represents a stack)."""
        # Using a set should give the same hash when we just move a sequence from one empty stack to another empty one
        state = set()
        state.add(len(self.deck.cards))
        for stack in self.stacks:
            stack_representation = tuple(card.encode() for card in stack.cards)
            state.add(stack_representation)

        return tuple(state)

    def get_hashed_state(self):
        return hash(self.get_state())

    def get_stack(self, stack_id: int) -> Stack:
        if stack_id >= 0 and stack_id < len(self.stacks):
            return self.stacks[stack_id]
        else:
            raise ValueError("Invalid stack id")

    def _count_breaking_cards(self, condition):
        count = 0
        for stack in self.stacks:
            for i in range(1, len(stack.visible_cards())):
                if condition(stack.visible_cards(), i):
                    count += 1
        return count

    def count_cards_breaking_stackable(self):
        return self._count_breaking_cards(
            lambda cards, i: not cards[i - 1].can_stack(cards[i])
        )

    def count_cards_breaking_sequence(self):
        return self._count_breaking_cards(
            lambda cards, i: not (
                cards[i - 1].can_stack(cards[i]) and cards[i - 1].same_suit(cards[i])
            )
        )

    def count_hidden_cards(self) -> int:
        """Count the number of hidden cards"""
        count = 0
        for stack in self.stacks:
            count += len(stack.hidden_cards())
        return count

    def count_visible_cards(self) -> int:
        count = 0
        for stack in self.stacks:
            count += len(stack.visible_cards())
        return count

    def count_empty_stacks(self) -> int:
        count = 0
        for stack in self.stacks:
            if stack.is_empty():
                count += 1
        return count

    def count_blocked_stacks(self) -> int:
        """How many stacks have a king on basement"""
        count = 0
        for stack in self.stacks:
            if stack.is_empty() or stack.first_visible_card != 0:
                continue
            if stack.cards[0].rank == 13:
                count += 1
        return count

    def count_semi_empty_stacks(self) -> int:
        """
        Count the number of stacks that can be emptied in a single move.

        A stack is considered 'semi-empty' if the sequence of visible cards
        on top of the stack can be legally moved to another stack, thus emptying it.

        :return: The number of semi-empty stacks.
        """
        semi_empty_count = 0

        for stack_index, stack in enumerate(self.stacks):
            if stack.is_empty():
                continue

            # Check if the entire visible sequence can be moved
            top_sequence_start = stack.first_accessible_sequence
            if top_sequence_start == 0 and self._get_valid_moves_to_other_stacks(
                stack_index, 0
            ):
                semi_empty_count += 1

        return semi_empty_count

    def count_completed_stacks(self) -> int:
        return len(self.completed_stacks)

    def stacks_sequence_lengths(self):
        sequences: list[int] = []
        for stack in self.stacks:
            sequences.append(stack.valid_sequence_length())

        return sequences

    def stacks_stacked_lengths(self):
        sequences: list[int] = []
        for stack in self.stacks:
            sequences.append(stack.valid_stacked_length())

        return sequences

    def visible_card_ranks(self):
        ranks = [0] * 13
        for stack in self.stacks:
            for card in stack.visible_cards():
                ranks[card.rank - 1] += 1

        return ranks

    def total_rank_stacked_cards(self) -> int:
        rank = 0
        for stack in self.stacks:
            for card in stack.get_stacked():
                rank += card.rank

        return rank

    def total_rank_sequence_cards(self) -> int:
        rank = 0
        for stack in self.stacks:
            for card in stack.get_sequence():
                rank += card.rank

        return rank

    def stacked_length_indicator(self) -> int:
        """
        Calculate a value based on the product of the highest card rank in each valid stacked sequence
        and the length of that sequence for each stack on the board.

        This function iterates through each stack on the board, examining the stacked cards.
        It sums up the products of the highest card rank in each valid stacked sequence and the length of that sequence,
        providing an indicator value for the board.

        The indicator helps in assessing the current state of the board by valuing longer stacked sequences more highly.

        :return: The sum of the products of the highest card ranks in stacked sequences and their lengths.
        """
        total_indicator = 0
        for stack in self.stacks:
            if stack.is_empty():
                continue

            cards = stack.visible_cards()
            current_rank = cards[0].rank
            current_sequence = 1
            for i in range(1, len(cards)):
                if cards[i - 1].can_stack(cards[i]):
                    current_sequence += 1
                else:
                    total_indicator += current_rank * current_sequence
                    current_sequence = 1
                    current_rank = cards[i].rank

            total_indicator += current_rank * current_sequence
        return total_indicator

    def sequence_length_indicator(self) -> int:
        """
        Calculate the sequence length indicator for each visible sequence in the stacks.

        This function iterates through each stack on the board, examining the visible
        cards. For each sequence of cards in the same suit that can be stacked on each other,
        it calculates the product of the sequence's length and the rank of the starting card.
        These products are then summed up to give a total indicator value for the board.

        The indicator helps in assessing the current state of the board by valuing longer
        sequences of the same suit more highly.

        :return: The sum of the products of sequence lengths and their starting card ranks.
        """
        total_indicator = 0
        for stack in self.stacks:
            if stack.is_empty():
                continue

            cards = stack.visible_cards()
            current_rank = cards[0].rank
            current_sequence = 1
            for i in range(1, len(cards)):
                if cards[i - 1].can_stack(cards[i]) and cards[i - 1].same_suit(
                    cards[i]
                ):
                    current_sequence += 1
                else:
                    total_indicator += current_rank * current_sequence
                    current_sequence = 1
                    current_rank = cards[i].rank

            total_indicator += current_rank * current_sequence
        return total_indicator

    def count_accessible_full_sequences(self) -> int:
        """
        Count the number of full sequences that can be made from readily accessible cards
        in the available stacked sequences on the board.

        A full sequence in Spider Solitaire typically involves a descending order
        from King to Ace of the same suit. This function iterates through each stack,
        starting from the first accessible card, and counts the number of such sequences.

        :return: The number of complete sequences available from accessible cards.
        """
        full_sequence_count = 0

        accessible_cards = self.get_accessible_cards()

        for suit in range(4):
            rank_count = [0] * 13

            for card in accessible_cards:
                if card.suit == suit:
                    rank_count[card.rank - 1] += 1

            if all(count >= 2 for count in rank_count):
                full_sequence_count += 2
            elif all(count >= 1 for count in rank_count):
                full_sequence_count += 1

        return full_sequence_count

    def get_accessible_cards(self) -> list[Card]:
        accessible_cards = []
        for stack in self.stacks:
            stacked_cards = stack.get_stacked()
            if stacked_cards:
                accessible_cards.extend(stacked_cards)
        return accessible_cards


class SpiderSolitaire:
    def __init__(self, seed=None):
        """Initialize the stacks: 4 stacks with 6 cards (last card face-up), and 6 stacks with 5 cards (last card face-up)"""
        self.board = Board(seed)
        self.move_count = 0
        self.just_completed_stack = False

    def move(self, from_stack, to_stack, card_index):
        """Move a sequence of cards from one stack to another"""
        if self.board.is_valid_move(from_stack, to_stack, card_index):
            self.board.move(from_stack, to_stack, card_index)

            self.move_count += 1

            if self.is_game_won():
                print("Congratulations! You won the game.")
                return
            if self.is_game_lost():
                print("No more moves available. Game lost.")

    def move_by_index(self, from_stack_index, to_stack_index, card_index):
        self.board.move_by_index(from_stack_index, to_stack_index, card_index)
        self.move_count += 1

        if self.is_game_won():
            print("Congratulations! You won the game.")
            return
        if self.is_game_lost():
            print("No more moves available. Game lost.")

    def move_pile_to_pile(self, from_stack_index, to_stack_index) -> bool:
        """Move a sequence of cards from one stack to another stack."""
        if from_stack_index < 0 or from_stack_index >= len(self.board.stacks):
            raise ValueError("Invalid from_stack_index")
        if to_stack_index < 0 or to_stack_index >= len(self.board.stacks):
            raise ValueError("Invalid to_stack_index")

        from_stack = self.board.stacks[from_stack_index]
        to_stack = self.board.stacks[to_stack_index]

        if to_stack.is_empty():
            card_index = from_stack.first_accessible_sequence
            self.move(from_stack, to_stack, card_index)
            return True

        # Iterate through the from_stack to find a valid sequence to move
        else:
            for card_index in range(len(from_stack.cards) - 1, -1, -1):
                if self.board.is_valid_move(from_stack, to_stack, card_index):
                    self.move(from_stack, to_stack, card_index)
                    return True

        return False

    def is_game_won(self):
        """Check if the game has been won"""
        return self.board.is_game_won()

    def is_game_lost(self):
        """Check if no more moves are available"""

        if self.move_count > 2000:
            return True

        return self.board.is_game_lost()

    def draw_from_deck(self):
        """Draw more cards from the deck"""
        drawn = self.board.draw_from_deck()
        if drawn:
            if self.is_game_won():
                print("Congratulations! You won the game.")

            if self.is_game_lost():
                print("No more moves available. Game lost.")

        return drawn

    def is_game_over(self):
        return self.is_game_lost() or self.is_game_won()

    def display_game_state(self):
        """Display the current state of the game"""
        self.board.display_game_state()


class SimpleSpiderSolitaire(SpiderSolitaire):
    def __init__(self, seed=None):
        """Initialize the stacks: 4 stacks with 6 cards (last card face-up), and 6 stacks with 5 cards (last card face-up)"""
        self.deck = SimpleDeck(seed)
        self.stacks = []
        self.completed_stacks = []
        self.move_count = 0
        self.just_completed_stack = False

        for _ in range(4):
            stack_cards = self.deck.draw(1)
            self.stacks.append(Stack(stack_cards))

        for _ in range(6):
            stack_cards = self.deck.draw(2)
            stack_cards += self.deck.draw(1)
            self.stacks.append(Stack(stack_cards))

        self.board = Board(deck=self.deck, stacks=self.stacks)
