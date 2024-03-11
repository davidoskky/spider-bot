import collections.abc

from deck import Card


class CardSequence(collections.abc.Sequence):
    """Sequence of cards of the same suit in decreasing order"""

    def __init__(self, sequence: list[Card], start_index: int) -> None:
        for prev_card, next_card in zip(sequence, sequence[1:]):
            if not prev_card.can_sequence(next_card):
                raise ValueError(f"The provided cards are not in sequence: {sequence}")
        self.cards = sequence
        self.start_index = start_index

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, index):
        return self.cards[index]

    def __repr__(self):
        return repr(self.cards)

    def drop_top_card(self):
        if self.cards:
            del self.cards[0]

    def drop_bottom_card(self):
        if self.cards:
            del self.cards[-1]

    def top_card(self):
        return self.cards[0]

    def bottom_card(self):
        return self.cards[-1]


class StackedSequence(collections.abc.Sequence):
    """Sequence of cards in decreasing order"""

    def __init__(self, sequence: list[Card], start_index: int) -> None:
        for prev_card, next_card in zip(sequence, sequence[1:]):
            if not prev_card.can_stack(next_card):
                raise ValueError(f"The provided cards are not in sequence: {sequence}")
        self.cards = sequence
        self.sequences = cards_to_sequences(sequence, start_index)
        self.start_index = start_index

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, index):
        return self.cards[index]

    def __repr__(self):
        return repr(self.cards)

    def drop_top_card(self):
        if self.cards:
            del self.cards[0]
            self.start_index += 1
            self.sequences[0].drop_top_card()
            if len(self.sequences[0]) == 0:
                del self.sequences[0]

    def drop_bottom_card(self):
        if self.cards:
            del self.cards[-1]
            self.sequences[-1].drop_bottom_card()
            if len(self.sequences[-1]) == 0:
                del self.sequences[-1]

    def top_card(self):
        return self.cards[0]

    def bottom_card(self):
        return self.cards[-1]


def cards_to_sequences(cards: list[Card], first_id: int = 0) -> list[CardSequence]:
    """Takes a list of cards and returns a list of CardSequence objects representing sequences."""
    sequences: list[CardSequence] = []
    if not cards:
        return sequences

    current_sequence_start_index = first_id
    current_sequence: list[Card] = [cards[0]]

    for i, card in enumerate(cards[1:], start=first_id + 1):
        if current_sequence[-1].can_sequence(card):
            current_sequence.append(card)
        else:
            sequences.append(
                CardSequence(
                    sequence=current_sequence, start_index=current_sequence_start_index
                )
            )
            current_sequence_start_index = i
            current_sequence = [card]

    if current_sequence:
        sequences.append(
            CardSequence(
                sequence=current_sequence, start_index=current_sequence_start_index
            )
        )

    return sequences
