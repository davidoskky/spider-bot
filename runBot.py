import cProfile
import pstats
import io
from tabnanny import verbose
from spiderSolitaireBot import SpiderSolitaireBot
from spiderSolitaire import SpiderSolitaire

bot = SpiderSolitaireBot(SpiderSolitaire())
# print(f"Game solveable {bot.gameSolvable()}")
# bot.play_bfs()
bot.play_heuristic(verbose=False)
