from spiderSolitaire import Board, InvalidMoveError, Move


class MoveEvaluator:
    def __init__(self, board: Board) -> None:
        self.initial_board = board
        self.reset()

    def reset(self):
        self.board = self.initial_board.clone()
        self.moves: list[Move] = []

    def move_possible(self, move: Move) -> bool:
        return self.moves_possible([move])

    def moves_possible(self, moves: list[Move]) -> bool:
        initial_board = self.board.clone()
        try:
            self.board.execute_moves(moves)
            self.moves.extend(moves)
            return True
        except InvalidMoveError:
            self.board = initial_board
            return False

    def get_moves(self) -> list[Move]:
        return self.moves
