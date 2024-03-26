from deck import Card, Deck, Rank, Suit
from spiderSolitaire import Board, Stack

'''
from util_tests import generate_board_from_string
board = generate_board_from_string("""Stack 0: XX XX XX XX XX 6♥ 5♣ 5♦ 4♦ 3♦ 12♥ 11♠ 3♥ 
Stack 1: XX 13♦ 12♦ 11♦ 12♦ 
Stack 2: XX XX 7♦ 6♥ 5♥ 4♥ 3♦ 
Stack 3: XX XX XX XX 1♣ 9♥ 8♣ 2♠ 1♠ 
Stack 4: XX 8♥ 13♣ 12♠ 
Stack 5: XX XX XX 13♣ 12♣ 11♣ 10♦ 9♠ 8♠ 
Stack 6: 6♦ 5♣ 4♣ 3♣ 2♣ 1♣ 
Stack 7: XX XX XX XX 2♥ 1♦ 1♥ 7♥ 
Stack 8: XX XX XX 13♥ 12♥ 11♦ 10♠ 9♠ 8♠ 9♦ 8♦ 
Stack 9: XX XX XX XX 5♠ 3♠ 2♠ 1♦ 2♦ 8♣ 7♠ 6♠ 
""")
'''


def parse_card(card_str: str) -> Card:
    """Parse a card string into a Card object."""
    suit_map = {"♠": Suit.SPADES, "♣": Suit.CLUBS, "♥": Suit.HEARTS, "♦": Suit.DIAMONDS}
    rank_map = {"A": Rank.ACE, "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING}

    if len(card_str) < 2 or card_str[-1] not in suit_map:
        raise ValueError(f"Invalid card format: {card_str}")

    rank_str, suit_str = card_str[:-1], card_str[-1]
    rank = rank_map.get(rank_str, Rank(int(rank_str)))
    suit = suit_map[suit_str]

    return Card(rank, suit)


def parse_stack(stack_str):
    """Parse a stack string into a list of Card objects."""
    card_strs = [card_str for card_str in stack_str.split(" ") if card_str.strip()]
    cards = []

    for card_str in card_strs:
        if card_str != "XX":
            cards.append(parse_card(card_str))
        else:
            cards.append(Card(Rank.ACE, Suit.SPADES))

    return cards


def generate_board_from_string(board_str):
    """Generate a Board object from a string representation."""
    stack_strs = board_str.strip().split("\n")
    stacks = []

    for stack_str in stack_strs:
        # Extract stack part after colon
        stack_list = stack_str.split(": ")
        if len(stack_list) < 2:
            stacks.append(Stack([]))
            continue

        stack_part = stack_list[1]
        cards = parse_stack(stack_part)
        stack = Stack(cards)
        stack.first_visible_card = next(
            (i for i, card in enumerate(stack_part.split(" ")) if card != "XX"), 0
        )
        stacks.append(stack)

    return Board(stacks=tuple(stacks), deck=Deck(), completed_stacks=0)
