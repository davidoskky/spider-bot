from moves_exploration import (
    DEFAULT_WEIGTHS,
    find_improved_equivalent_position,
    find_progressive_actions,
    is_board_winnable,
    score_board,
)
from spiderSolitaire import SpiderSolitaire


class SpiderSolitaireBot:
    def __init__(self, game: SpiderSolitaire):
        self.game = game

    def evaluate_move(self, move):
        """Evaluate and score the move based on a simple heuristic"""
        from_stack, to_stack, card_index = move
        score = 0

        # Higher score for moves that uncover a hidden card
        if card_index > 0 and not from_stack.cards[card_index - 1].face_up:
            score += 5

        # Higher score for moving to an empty stack
        if to_stack.is_empty():
            score += 3

        return score

    def select_best_move(self):
        """Select the best move based on the evaluation"""
        available_moves = self.game.board.list_available_moves()

        # Filter out moves that are not between stacks
        stack_moves = [move for move in available_moves if len(move) == 3]

        # Avoid repeating the same move
        stack_moves = [
            move for move in stack_moves if move not in self.move_history[-4:]
        ]

        if not stack_moves:
            return "draw_from_deck"

        return max(stack_moves, key=self.evaluate_move)

    def make_move(self):
        """Make the best move according to the bot's strategy"""
        best_move = self.select_best_move()

        if best_move == "draw_from_deck":
            self.game.draw_from_deck()
            self.move_history = []
        else:
            from_index, to_index, card_index = best_move
            self.game.move(from_index, to_index, card_index)
            self.move_history.append(best_move)

    def select_best_progressive_action(self, weights=DEFAULT_WEIGTHS):
        """
        Select the best progressive action based on the scoring of resulting board states.

        :param board: The current state of the Spider Solitaire game.
        :param weights: A dictionary of weights for different scoring criteria.
        :return: The best move (action) to take.
        """
        best_score = float("-inf")
        best_path = None

        for path in find_progressive_actions(self.game.board):
            # Simulate the move
            simulated_board = self.game.board.clone()
            for move in path:
                simulated_board.move_by_index(*move)

            # Score the resulting board state
            score = score_board(simulated_board, weights)

            # Update the best move if this move has a higher score
            if score > best_score:
                best_score = score
                best_path = path

        return best_path

    def play(self):
        """Play the game until it is won or lost"""
        while not self.game.is_game_won() and not self.game.is_game_lost():
            self.make_move()

        print(f"Game finished after {self.game.move_count} moves.")
        if self.game.is_game_lost():
            print(f"Game lost")
            self.game.display_game_state()

    def play_bfs(self):
        moves = True
        while moves:
            self.game.display_game_state()
            while find_improved_equivalent_position(self.game.board):
                self._execute_moves(find_improved_equivalent_position(self.game.board))
            self._execute_moves(find_progressive_actions(self.game.board))
            moves = find_improved_equivalent_position(
                self.game.board
            ) + find_progressive_actions(self.game.board)

    def play_heuristic(self, weights=DEFAULT_WEIGTHS):
        moves = True
        while moves:
            self.game.display_game_state()
            moves = False
            improved_position_moves = find_improved_equivalent_position(self.game.board)
            while improved_position_moves:
                moves = True
                # print("Improve equivalent")
                self._execute_moves(improved_position_moves)
                improved_position_moves = find_improved_equivalent_position(
                    self.game.board
                )
            progressive_path = self.select_best_progressive_action(weights)
            if progressive_path:
                moves = True
                # print("Progressive")
                for move in progressive_path:
                    self.game.move_by_index(*move)

            if moves == False and len(self.game.board.deck.cards) > 0:
                moves = True
                self.game.draw_from_deck()
            print("Completed cycle")

    def gameSolvable(self) -> bool:
        return is_board_winnable(self.game.board)

    def _execute_moves(self, moves):
        if moves:
            for move in moves[0]:
                self.game.move_by_index(*move)
                # print(f"from {move[0]}, to {move[1]}")
