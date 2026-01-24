import numpy as np


class EquivalentBoards:
    """Specifies simmetry axes and rotations of a board game."""

    def __init__(
        self,
        rotate90: bool,
        rotate180: bool,
        rotate270: bool,
        reflect_horizontal: bool,
        reflect_vertical: bool,
        reflect_diag_nw_se: bool,
        reflect_diag_ne_sw: bool,
    ):
        self.rotate90 = rotate90
        self.rotate180 = rotate180
        self.rotate270 = rotate270
        self.reflect_horizontal = reflect_horizontal
        self.reflect_vertical = reflect_vertical
        self.reflect_diag_nw_se = reflect_diag_nw_se
        self.reflect_diag_ne_sw = reflect_diag_ne_sw

    def get_equivalent_boards(self, board: np.ndarray) -> list[np.ndarray]:
        """Generates all unique equivalent boards applying the specified symmetries.

        Args:
            board: original board as numpy array.

        Returns:
            list of equivalent boards as numpy arrays.
        """
        transformations = [identity]
        if self.rotate90:
            transformations.append(rotate90)
        if self.rotate180:
            transformations.append(rotate180)
        if self.rotate270:
            transformations.append(rotate270)
        if self.reflect_horizontal:
            transformations.append(reflect_horizontal)
        if self.reflect_vertical:
            transformations.append(reflect_vertical)
        if self.reflect_diag_nw_se:
            transformations.append(reflect_diag_nw_se)
        if self.reflect_diag_ne_sw:
            transformations.append(reflect_diag_ne_sw)

        equivalent_boards = []
        for transform in transformations:
            transformed_board = transform(board)
            if not any(np.array_equal(transformed_board, b) for b in equivalent_boards):
                equivalent_boards.append(transformed_board)
        return equivalent_boards


def identity(board):
    """Returns the board unchanged."""
    return board.copy()


def rotate90(board):  # Clockwise 90°
    """Rotates the board 90 degrees clockwise."""
    return np.rot90(board, k=-1)


def rotate180(board):
    """Rotates the board 180 degrees."""
    return np.rot90(board, k=-2)


def rotate270(board):
    """Rotates the board 270 degrees clockwise."""
    return np.rot90(board, k=-3)


def reflect_horizontal(board):
    """Flips the board upside-down."""
    return np.flipud(board)


def reflect_vertical(board):
    """Flips the board left-right."""
    return np.fliplr(board)


def reflect_diag_nw_se(board):
    """Reflects the board along the main diagonal (top-left to bottom-right)."""
    return np.transpose(board)


def reflect_diag_ne_sw(board):
    """Reflects the board along the anti-diagonal (top-right to bottom-left)."""
    return np.fliplr(np.flipud(np.transpose(board)))
