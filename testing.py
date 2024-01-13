from deck import Card
from spiderSolitaire import SpiderSolitaire
from spiderSolitaire import Stack

game = SpiderSolitaire()

game.display_game_state()
print(repr(game.board.get_state()))

game.draw_from_deck()
print(repr(game.board.get_state()))

stacks = []
for i in range(10):
    empty_stack = Stack([])
    stacks.append(empty_stack)

single_card_stack = Stack([Card(2, 2)])

first_stack = stacks.copy()
first_stack[0] = single_card_stack
second_stack = stacks.copy()
second_stack[1] = single_card_stack

game.board.stacks = tuple(first_stack)
print(repr(game.board.get_state()))

game.board.stacks = tuple(second_stack)
print(repr(game.board.get_state()))
