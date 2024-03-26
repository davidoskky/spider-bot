import random
from dataclasses import dataclass
from enum import Enum, IntEnum, auto


class Suit(Enum):
    SPADES = auto()
    CLUBS = auto()
    HEARTS = auto()
    DIAMONDS = auto()

    def __str__(self):
        return "♠♣♥♦"[self.value - 1]


class Rank(IntEnum):
    ACE = 1
    TWO = auto()
    THREE = auto()
    FOUR = auto()
    FIVE = auto()
    SIX = auto()
    SEVEN = auto()
    EIGHT = auto()
    NINE = auto()
    TEN = auto()
    JACK = auto()
    QUEEN = auto()
    KING = auto()

    def __str__(self):
        special_ranks = {1: "A", 11: "J", 12: "Q", 13: "K"}
        return special_ranks.get(self.value, str(self.value))

    @property
    def is_ace(self):
        return self == Rank.ACE

    @property
    def is_king(self):
        return self == Rank.KING


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    def __repr__(self):
        return f"{self.rank}{self.suit}"

    def encode(self):
        """Encode a card into a unique identifier."""
        return self.suit.value * 13 + self.rank

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
                Card(rank, suit) for suit in Suit for rank in Rank for _ in range(2)
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
