# Degrees of Freedom

In this document I will outline the concept of degrees of freedom and how it can be used to predict which moves can be performed or not without having to simulate the full chain of moves.

The degrees of freedom translate into how many cards in the same stack can be moved reversibly by just stacking them onto empty stacks, in this definition the amount of cards which can be stacked onto other cards present on the board is not considered.
As such, the number of degrees of freedom solely depends on the amount of free stacks present on the board.
The number of degrees of freedom of the board can be rapidly calculated with the formula $DoF = 2^n - 1$ where n is the amount of empty stacks currently present on the board.

Since the board can be rearranged in several different ways through reversible moves, and each of those states can be translated to any other state through reversible moves, it is also useful to define the maximum attainable amount of degrees of freedom of a reversible state.
This is done by identifying the maximum amount of empty stacks which can be attained through reversible moves and calculating the amount of degrees of freedom with the previous formula.

The first simplification which can be obtained by using this identifier is sequences of moves coming from one single stack which are always impossible.
If we are trying to move a stack composed of 3 sequences of cards of the same rank and none of the cards in those sequences can be placed onto an other card in the board, we need at least 3 degrees of freedom in the current state. This translates into having 2 empty stacks on the board.
If we have less than 2 empty stacks, it is certain that moving the cards is impossible through a sequence of reversible moves.

If it is possible to increase the degrees of freedom of the board through reversible moves to a number which is larger or equal to the amount of sequences which we wish to move, then we need to consider 2 cases:

1. It is possible to free enough stacks without increasing the amount of sequences to move in the stack we wish to move sequences from:
    In this case the move is possible.
2. It is not possible to free enough stacks withouth increasing the amount of sequences to move in the stack we wish to move sequences from:
    In this case we need to perform the calculation of required degrees of freedom and available degrees of freedom again. If the available degrees of freedom are larger or equal to the required ones, then the move is possible.

This is the simplest case, the complexity increases if some of the cards in the set of sequences we are trying to move can be stacked on top of other cards present on the board.
In such case two different algorithms are plausible.

1. Use the first plausible card:
    - Evaluate which one is the first card which can be stacked on top of another card.
    - If the required degrees of freedom is large enough, reset the counter of required degrees of freedom.
    - Remove the destination stack from the stacks we are considering for stacking further cards.
    - Repeat the steps until we moved the card we wished to move.

2. Use the card which requires most degrees of freedom to be moved while still being movable.
    - Identify all cards in the sequences which we wish to move which can be stacked on top of another card on the board.
    - Evaluate the required degrees of freedom to move each of them.
    - If several cards have the same amount of degrees of freedom, only consider the one which has the highest rank.
    - Remove the destination stack from the stacks we are considering for stacking further cards.
    - Repeat the steps until we moved the card we wished to move.

Problems:

- These algorithms do not consider freeing further stacks if doing so is required or advantageous.
- These algorithms may split a sequence of cards of the same seed even when it is not strictly required to perform the move, this may result into a larger amount of moves than optimal.

These algorithms only consider moving cards away from a single stack to an empty stack. However, two other cases are often required:

1. The destination stack is not a free stack.
    In this case the amount of required degrees of freedom should be decreased by 1.
2. We are switching cards between two stacks.
    In this case we need to move cards away from both stacks and the calculation of degrees of freedom in the previously outlined way does not allow determining how many empty stacks are required to move two distinct sequences of cards.
    A clear algorithm to determine this number still has to be outlined and would be of great use since this kind of moves is very often present in the game.
    Such algorithm should apply when none of the cards can be placed on top of another card, as well as when they can be placed on top of other cards.


Conclusion:

Currently, we have a clearly defined algorithm which identifies the amount of empty stacks required to move sequences of cards from a single stack.
We have also some tentative algorithms to identify how many empty stacks are required when some of the cards in those sequences can be placed on top of another card in the board, however such algorithms are not optimized and may fail in cases when additional degrees of freedom are required and may be obtained.
We have no defined algorithm to move sequences of cards from more than one stack.
