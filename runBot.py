import cProfile
import io
import pstats
from tabnanny import verbose

from spiderSolitaire import SpiderSolitaire
from spiderSolitaireBot import SpiderSolitaireBot

bot = SpiderSolitaireBot(SpiderSolitaire(seed=66))
# print(f"Game solveable {bot.gameSolvable()}")
# bot.play_bfs()
bot.play_heuristic(verbose=1)

for i in range(1000):
    bot = SpiderSolitaireBot(SpiderSolitaire(seed=i))
    bot.play_heuristic(verbose=0)
    if bot.game.is_game_won():
        print(f"Game won, seed: {i}")
