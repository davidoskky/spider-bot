import pygad

from moves_exploration import DEFAULT_WEIGTHS
from spiderSolitaire import SpiderSolitaire
from spiderSolitaireBot import SpiderSolitaireBot


def fitness_func(self, solution, solution_idx):
    print(f"Testing New Solution {solution}")
    decoded_weights = decode_solution(solution)
    total_steps = 0
    num_games = 10

    for i in range(num_games):
        print(f"Game {i}")
        bot = SpiderSolitaireBot(SpiderSolitaire())
        bot.play_heuristic(decoded_weights)
        total_steps += bot.game.move_count

    average_steps = total_steps / num_games
    print(f"Average steps: {average_steps}")
    return average_steps


def decode_solution(solution):
    return {key: value for key, value in zip(DEFAULT_WEIGTHS.keys(), solution)}


ga_instance = pygad.GA(
    num_generations=500,
    num_parents_mating=4,
    fitness_func=fitness_func,
    sol_per_pop=10,
    num_genes=len(DEFAULT_WEIGTHS),
    init_range_low=-500,
    init_range_high=500,
    mutation_percent_genes=5,
)

ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
optimized_weights = decode_solution(solution)
print("Optimized Weights:", optimized_weights)
print("Fitness of the best solution:", solution_fitness)
