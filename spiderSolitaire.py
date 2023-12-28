from deck import Deck, SimpleDeck, Card
import copy
from moves_exploration import find_progressive_actions


class Stack:
    def __init__(self, cards):
        self.cards = cards
        self.first_visible_card = len(cards) - 1

    def __repr__(self):
        representation = ""
        for i, card in enumerate(self.cards):
            if i < self.first_visible_card:
                representation += "XX "
            else:
                representation += repr(card) + " "

        return representation

    def clone(self):
        cloned_stack = Stack(copy.copy(self.cards))
        cloned_stack.first_visible_card = self.first_visible_card
        return cloned_stack

    def is_visible(self, card_index: int) -> bool:
        """Check if the card at card_index is face-up (visible)."""
        return card_index >= self.first_visible_card

    def can_stack(self, card: Card) -> bool:
        return self.is_empty() or self.top_card().can_stack(card)

    def flip_top_card(self):
        """Flip the top card to face-up."""
        if self.cards and self.first_visible_card >= len(self.cards):
            self.first_visible_card = len(self.cards) - 1

    def is_valid_sequence(self, start_index) -> bool:
        """Check if a sequence starting from start_index is valid."""
        if not self.is_visible(start_index):
            return False

        return all(
            self.cards[i].can_stack(self.cards[i + 1])
            and self.cards[i].same_suit(self.cards[i + 1])
            for i in range(start_index, len(self.cards) - 1)
        )

    def is_valid_stacked(self, start_index):
        """Check if a sequence starting from start_index is valid"""
        if self.is_visible(start_index):
            return False

        return all(
            self.cards[i].can_stack(self.cards[i + 1])
            for i in range(start_index, len(self.cards) - 1)
        )

    def visible_cards(self):
        return self.cards[self.first_visible_card :]

    def hidden_cards(self):
        return self.cards[: self.first_visible_card]

    def add_sequence(self, sequence):
        """Add a sequence of cards to the stack"""
        self.cards.extend(sequence)

    def remove_sequence(self, start_index):
        if not self.is_visible(start_index):
            raise ValueError("Cannot remove a hidden sequence")
        sequence = self.cards[start_index:]
        self.cards = self.cards[:start_index]
        self.flip_top_card()
        return sequence

    def top_card(self):
        return self.cards[-1] if self.cards else None

    def is_empty(self):
        return not self.cards

    def first_card_of_valid_sequence(self) -> int:
        """Find the index of the first card of the valid sequence from the top."""
        for i in range(self.first_visible_card, len(self.cards)):
            if self.is_valid_sequence(i):
                return i
        return 0

    def valid_sequence_length(self) -> int:
        return len(self.cards) - self.first_card_of_valid_sequence()

    def first_card_not_in_sequence(self):
        """Find the index of the first card that is not in sequence from the top."""
        i = self.first_card_of_valid_sequence()
        if i > 0:
            return i - 1
        else:
            return None

    def get_sequence(self):
        return self.cards[self.first_card_of_valid_sequence() :]

    def first_card_of_valid_stacked(self) -> int:
        """Find the index of the first card of the valid sequence from the top."""
        for i in range(len(self.cards)):
            if self.is_valid_stacked(i):
                return i
        return 0

    def valid_stacked_length(self) -> int:
        return len(self.cards) - self.first_card_of_valid_stacked()

    def get_stacked(self):
        return self.cards[self.first_card_of_valid_stacked() :]


class Board:
    INITIAL_STACKS_COUNT = 10
    CARDS_IN_SMALL_STACK = 5
    CARDS_IN_LARGE_STACK = 6

    def __init__(self, seed=None, stacks=None, deck=None) -> None:
        if stacks is not None and deck is not None:
            self._initialize_for_cloning(stacks, deck)
        else:
            self._initialize_new_game(seed)

    def _initialize_for_cloning(self, stacks: list[Stack], deck):
        self.deck = deck.clone()
        self.stacks = [stack.clone() for stack in stacks]
        self.completed_stacks = []

    def _initialize_new_game(self, seed):
        self.deck = Deck(seed)
        self.stacks: list[Stack] = []
        self.completed_stacks = []

        for i in range(Board.INITIAL_STACKS_COUNT):
            num_face_down = (
                Board.CARDS_IN_LARGE_STACK if i < 4 else Board.CARDS_IN_SMALL_STACK
            )
            stack_cards = self.deck.draw(num_face_down)
            stack = Stack(stack_cards)

            self.stacks.append(stack)

    def clone(self):
        """Create a new Board object with a deep clone of the card disposition."""
        cloned_stacks = [stack.clone() for stack in self.stacks]
        return Board(stacks=cloned_stacks, deck=self.deck.clone())

    def is_valid_move(self, from_stack: Stack, to_stack: Stack, card_index: int):
        """Check if moving a sequence of cards from one stack to another is valid."""
        if from_stack.is_empty():
            return False

        # Optimization: Do not allow moving a card on an empty stack to another empty stack
        # This check is more efficient as it does not require sequence validation
        if card_index == 0 and to_stack.is_empty():
            return False

        # Check if the sequence starting at 'card_index' is valid
        # This is more computationally intensive, so it's done after simpler checks
        if not from_stack.is_valid_sequence(card_index):
            return False

        return to_stack.can_stack(from_stack.cards[card_index])

    def move(self, source_stack: Stack, destination_stack: Stack, card_index: int):
        """Move a sequence of cards from one stack to another"""
        if not self.is_valid_move(source_stack, destination_stack, card_index):
            source_id = self.stacks.index(source_stack)
            destination_id = self.stacks.index(destination_stack)
            self.display_game_state()
            raise ValueError(
                f"Invalid move attempted. From {source_id}, To {destination_id}, card {card_index}"
            )
        sequence_to_move = source_stack.remove_sequence(card_index)
        destination_stack.add_sequence(sequence_to_move)
        self.check_for_completion(destination_stack)

    def move_by_index(self, source_stack_id, destination_stack_id, card_index):
        """Move a sequence of cards from one stack to another"""
        self.move(
            self.stacks[source_stack_id], self.stacks[destination_stack_id], card_index
        )

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

        return not any(
            self.is_valid_move(from_stack, to_stack, card_index)
            for i, from_stack in enumerate(self.stacks)
            for j, to_stack in enumerate(self.stacks)
            if i != j
            for card_index in range(len(from_stack.cards))
        )

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

    def check_for_completion(self, stack):
        """Check and update for any completed stacks in the given stack"""
        self.just_completed_stack = False
        if len(stack.cards) >= 13 and stack.is_valid_sequence(len(stack.cards) - 13):
            completed_sequence = stack.remove_sequence(len(stack.cards) - 13)
            self.completed_stacks.append(completed_sequence)
            self.just_completed_stack = True
            print("Completed sequence moved to the completed stacks.")

    def list_available_moves(self):
        """List all available moves in the current game state."""
        available_moves = self._list_stack_to_stack_moves() + self._list_deck_moves()
        return available_moves

    def _list_stack_to_stack_moves(self):
        """List all valid stack-to-stack moves."""
        moves = []
        for i, from_stack in enumerate(self.stacks):
            for card_index in range(
                from_stack.first_card_of_valid_sequence(), len(from_stack.cards)
            ):
                moves.extend(self._get_valid_moves_to_other_stacks(i, card_index))
        return moves

    def _get_valid_moves_to_other_stacks(self, from_index, card_index):
        """Get all valid moves of one cart in one stack to other stacks."""
        return [
            (from_index, to_index, card_index)
            for to_index, to_stack in enumerate(self.stacks)
            if from_index != to_index
            and self.is_valid_move(self.stacks[from_index], to_stack, card_index)
        ]

    def _list_deck_moves(self):
        """List moves related to the deck, if any."""
        return [("draw_from_deck",)] if self.deck.cards else []

    def display_game_state(self):
        """Display the current state of the game"""
        for i, stack in enumerate(self.stacks):
            print(f"Stack {i}: ", end="")
            print(stack)
        print("Completed Stacks: ", len(self.completed_stacks))

    def get_state(self):
        """Return the current game state as a list of lists (each list represents a stack)."""
        state = []
        state.append(len(self.deck.cards))
        for stack in self.stacks:
            stack_representation = tuple(card.encode() for card in stack.cards)
            state.append(stack_representation)

        return tuple(state)

    def get_hashed_state(self):
        return hash(self.get_state())

    def count_cards_breaking_stackable(self) -> int:
        """Count the number of cards that break a sequence without considering suit"""
        count = 0
        for stack in self.stacks:
            cards = stack.visible_cards()
            for i in range(1, len(cards)):
                if not cards[i - 1].can_stack(cards[i]):
                    count += 1
        return count

    def count_cards_breaking_sequence(self) -> int:
        """Count the number of cards that break a sequence without considering suit"""
        count = 0
        for stack in self.stacks:
            cards = stack.visible_cards()
            for i in range(1, len(cards)):
                if not (
                    cards[i - 1].can_stack(cards[i]) or cards[i - 1].same_suit(cards[i])
                ):
                    count += 1
        return count

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
            # Special logic for moving to an empty pile
            # Move the longest valid sequence or any card
            card_index = from_stack.first_card_of_valid_sequence()
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
