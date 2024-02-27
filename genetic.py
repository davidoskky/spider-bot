import random
from random import randint

import pygad

from moves_exploration import DEFAULT_WEIGHTS
from spiderSolitaire import SpiderSolitaire
from spiderSolitaireBot import SpiderSolitaireBot

WINNABLE_GAMES = [
    9,
    23,
    66,
    77,
    117,
    127,
    137,
    139,
    172,
    185,
    215,
    220,
    240,
    266,
    284,
    332,
    345,
    359,
    361,
    364,
    372,
    386,
    432,
    435,
    454,
    481,
    491,
    527,
    630,
    632,
    648,
    701,
    730,
    751,
    758,
    766,
    853,
    869,
    874,
    897,
    900,
    909,
    912,
    959,
    967,
    980,
]


def fitness_func(self, solution, solution_idx):
    print(f"Testing New Solution {solution}")
    decoded_weights = decode_solution(solution)
    total_steps = 0
    total_won = 0
    total_completed_stacks = 0
    num_games = 50

    for i, seed in enumerate(WINNABLE_GAMES):
        # print(f"Game {i}")
        bot = SpiderSolitaireBot(SpiderSolitaire(seed=seed))
        bot.play_heuristic(decoded_weights, verbose=0)
        total_steps += bot.game.move_count
        if bot.game.is_game_won():
            total_won += 1

        total_completed_stacks += len(bot.game.board.completed_stacks)

    average_steps = total_steps / num_games
    average_completed = total_completed_stacks / num_games
    average_won = total_won / num_games
    print(f"Average steps: {average_steps}")
    print(f"Average completed: {average_completed}")
    print(f"Average won: {average_won}")
    return average_completed


def decode_solution(solution):
    return {key: value for key, value in zip(DEFAULT_WEIGHTS.keys(), solution)}


if __name__ == "__main__":

    def_values = [item[1] for item in DEFAULT_WEIGHTS.items()]

    initial_population = [
        def_values,
        random.sample(range(-500, 500), len(DEFAULT_WEIGHTS)),
        random.sample(range(-500, 500), len(DEFAULT_WEIGHTS)),
        random.sample(range(-500, 500), len(DEFAULT_WEIGHTS)),
    ]

    ga_instance = pygad.GA(
        num_generations=15,
        num_parents_mating=4,
        initial_population=initial_population,
        fitness_func=fitness_func,
        sol_per_pop=15,
        num_genes=len(DEFAULT_WEIGHTS),
        init_range_low=-500,
        init_range_high=500,
        mutation_percent_genes=5,
    )

    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    optimized_weights = decode_solution(solution)
    print("Optimized Weights:", optimized_weights)
    print("Fitness of the best solution:", solution_fitness)
