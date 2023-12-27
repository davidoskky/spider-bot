import cProfile
import pstats
import io
from spiderSolitaireBot import SpiderSolitaireBot
from spiderSolitaire import SpiderSolitaire

profiler = cProfile.Profile()

bot = SpiderSolitaireBot(SpiderSolitaire())
# print(f"Game solveable {bot.gameSolvable()}")
# bot.play_bfs()
bot.play_heuristic()
