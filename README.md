# Spider Solitaire Bot

This is a work in progress project.
The ultimate objective of this project is identifying an optimal strategy to play the Spider Solitaire game which allows the highest rate of victories.

To reach this objective, the strategy is fully implemented in the code present in this repository.
There are no plans of using Artificial Intelligence to identify such strategy, as such the code can be interpreted and translated into a human understandable algorithm for playing.

The Spider game is implemented in this repository, it is composed of 4 classes: Board, Stack, Card and Deck. These are contained in the modules spiderSolitaire and deck.
The Bot strategy is contained in the module spiderSolitaireBot, while the mechanisms to perform moves are contained in the module moves_exploration.

The whole codebase is currently extremely disorganized. A major refactoring will occur once I develop the strategy well enough to win somewhat reliably.

At the current state of things, the bot is able to play a game and is quite decent at performing reversible moves.
Problems arise in the choice of non-reversible moves, which is where strategy matters the most.
A single game may take up to half an hour to be played until no more moves are available. This is expected to improve through several optimization strategies I'm implementing.

The bot has been observed to win the game in one occasion. Currently, it is generally able to complete 1 or 2 stacks out of 8 in many games it plays.
