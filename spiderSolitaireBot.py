from moves_exploration import (
    DEFAULT_WEIGTHS,
    Move,
    find_improved_equivalent_position,
    find_moves_freeing_covered_cards,
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
        best_path, best_score = self.select_best_scoring_path(
            find_progressive_actions(self.game.board), weights
        )

        return best_path, best_score

    def select_best_scoring_path(
        self, paths: list[list[Move]], weights=DEFAULT_WEIGTHS
    ):
        best_score = float("-inf")
        best_path = []

        for path in paths:
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

        return best_path, best_score

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
        current_score = score_board(self.game.board, weights)
        cycle = 0

        while True:
            self.game.display_game_state()
            moves_made = False

            # Execute all improved equivalent positions
            improved_position_moves = find_improved_equivalent_position(self.game.board)
            if improved_position_moves:
                self._execute_moves(improved_position_moves)
                moves_made = True

            # If no improved equivalent positions were made, find and score other moves
            if not moves_made:
                all_other_moves, score = self.select_best_scoring_path(
                    find_moves_freeing_covered_cards(self.game.board), weights
                )
                if all_other_moves and current_score < score:
                    for move in all_other_moves:
                        self.game.board.move_by_index(*move)
                    current_score = score
                    moves_made = True

            # If no moves were made so far, try progressive moves
            if not moves_made:
                progressive_path, score = self.select_best_progressive_action(weights)
                if progressive_path:
                    for move in progressive_path:
                        self.game.board.move_by_index(*move)
                    current_score = score
                    moves_made = True

            # If still no moves were made and cards are in the deck, draw from deck
            if not moves_made and len(self.game.board.deck.cards) > 0:
                self.game.board.draw_from_deck()
                current_score = score_board(self.game.board, weights)
                moves_made = True

            if not moves_made:
                print("No moves available.")
                break

            cycle += 1
            print(f"Completed cycle {cycle}")

    def play_heuristic_old(self, weights=DEFAULT_WEIGTHS):
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

            # TODO: Use the Heuristic for this
            if moves == False:
                other_moves = find_moves_freeing_covered_cards(self.game.board)
                if other_moves:
                    self._execute_moves(other_moves)
                    moves = True

            if moves == False and len(self.game.board.deck.cards) > 0:
                moves = True
                self.game.draw_from_deck()
            print("Completed cycle")

    def gameSolvable(self) -> bool:
        return is_board_winnable(self.game.board)

    def _execute_moves(self, moves):
        if moves:
            for move in moves[0]:
                self.game.board.move_by_index(*move)
                # print(f"from {move[0]}, to {move[1]}")
