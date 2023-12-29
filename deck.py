import random
from collections import namedtuple

SUITS = "♠♣♥♦"


class Card:
    def __init__(self, rank: int, suit):
        self.rank = rank
        self.suit = suit

    def clone(self):
        """Create a deep clone of this card."""
        return Card(self.rank, self.suit)

    def __repr__(self):
        return f"{self.rank}{SUITS[self.suit]}"

    def encode(self):
        """Encode a card into a unique identifier."""
        return self.suit * 13 + self.rank

    def can_stack(self, card: "Card") -> bool:
        """Check if the given card can be stacked on this card"""
        return self.rank == (card.rank + 1)

    def can_sequence(self, card: "Card") -> bool:
        return self.can_stack(card) and self.same_suit(card)

    def same_suit(self, card: "Card") -> bool:
        """Check if this card has the same suit as the given card"""
        return self.suit == card.suit


class Deck:
    def __init__(self, seed=None, cards: list[Card] | None = None):
        if cards is None:
            self.cards = [
                Card(rank, suit)
                for suit in range(4)
                for rank in range(1, 14)
                for _ in range(2)  # 2 decks
            ]
            if seed:
                random.seed(seed)
            random.shuffle(self.cards)
        else:
            self.cards = cards

    def clone(self):
        return Deck(cards=self.cards)

    def draw(self, count):
        """Draw 'count' number of cards from the deck"""
        drawn_cards = self.cards[-count:]
        self.cards = self.cards[:-count]
        return drawn_cards


class SimpleDeck(Deck):
    def __init__(self, seed=None):
        self.cards = [
            Card(rank, suit)
            for suit in range(2)
            for rank in range(1, 14)
            for _ in range(1)  # 2 decks
        ]
        if seed:
            random.seed = seed
        random.shuffle(self.cards)
