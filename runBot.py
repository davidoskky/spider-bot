import cProfile
import io
import pstats
from tabnanny import verbose

from spiderSolitaire import SpiderSolitaire
from spiderSolitaireBot import SpiderSolitaireBot

bot = SpiderSolitaireBot(SpiderSolitaire())
# print(f"Game solveable {bot.gameSolvable()}")
# bot.play_bfs()
bot.play_heuristic(verbose=1)
