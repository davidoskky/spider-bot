import random
from collections import namedtuple

SUITS = "♠♣♥♦"


class Card:
    def __init__(self, rank: int, suit, face_up=False):
        self.rank = rank
        self.suit = suit
        self.face_up = face_up

    def clone(self):
        """Create a deep clone of this card."""
        return Card(self.rank, self.suit, self.face_up)

    def __repr__(self):
        return f"{self.rank}{SUITS[self.suit]}" if self.face_up else "XX"

    def encode(self):
        """Encode a card into a unique identifier."""
        if not self.face_up:
            return 0
        return self.suit * 13 + self.rank

    def can_stack(self, card: "Card") -> bool:
        """Check if the given card can be stacked on this card"""
        return self.rank == (card.rank + 1)

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

    def draw(self, count, face_up=False):
        """Draw 'count' number of cards from the deck"""
        drawn_cards = self.cards[-count:]
        self.cards = self.cards[:-count]
        for card in drawn_cards:
            card.face_up = face_up
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
