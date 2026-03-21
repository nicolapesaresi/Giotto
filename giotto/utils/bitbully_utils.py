import bitbully as bb


def convert_board(moves: list[int]) -> bb.Board:
    """Converts Connect4 moves list from giotto format to bitbully format."""
    board = bb.Board()

    if moves:
        for move in moves:
            board.play(move - 1)

    return board
