from spiderSolitaire import SpiderSolitaire

game = SpiderSolitaire()

game.display_game_state()
print(repr(game.get_state()))

game.draw_from_deck()
print(repr(game.get_state()))
