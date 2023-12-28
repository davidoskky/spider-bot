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
            self.game.display_game_state()
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
