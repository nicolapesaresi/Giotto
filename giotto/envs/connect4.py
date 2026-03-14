import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from giotto.envs.generic import GenericEnv
from giotto.utils.simmetries import EquivalentBoards


class Connect4Env(GenericEnv):
    """Connect4 environment."""

    simmetries = EquivalentBoards(
        rotate90=False,
        rotate180=False,
        rotate270=False,
        reflect_horizontal=False,
        reflect_vertical=True,
        reflect_diag_nw_se=False,
        reflect_diag_ne_sw=False,
    )

    def __init__(self):
        """Instantiates environment."""
        signs = ["o", "x", -1]  # third is empty place, accessed with -1
        rows = 6
        cols = 7
        super().__init__(signs, rows, cols)
        self.reset()

    def check_win(self, player_idx: int) -> bool:
        """Checks if the given player has won."""
        player_mask = self.board == player_idx

        # Horizontal: windows of 4 along columns for each row
        horiz_windows = sliding_window_view(player_mask, window_shape=(1, 4)).squeeze(axis=2)  # (rows, cols-3, 4)
        if np.any(np.all(horiz_windows, axis=-1)):
            return True

        # Vertical: windows of 4 along rows for each column
        vert_windows = sliding_window_view(player_mask, window_shape=(4, 1)).squeeze(axis=3)  # (rows-3, cols, 4)
        if np.any(np.all(vert_windows, axis=-1)):
            return True

        # Diagonal (\): windows of 4x4, check main diagonal
        square_windows = sliding_window_view(player_mask, window_shape=(4, 4))  # (rows-3, cols-3, 4, 4)
        diag_down_right = np.diagonal(square_windows, axis1=-2, axis2=-1)  # (rows-3, cols-3, 4)
        if np.any(np.all(diag_down_right, axis=-1)):
            return True

        # Diagonal (/): same 4x4 windows, check anti-diagonal by flipping columns
        flipped_mask = player_mask[:, ::-1]
        square_windows_flipped = sliding_window_view(flipped_mask, window_shape=(4, 4))
        diag_up_right = np.diagonal(square_windows_flipped, axis1=-2, axis2=-1)
        return bool(np.any(np.all(diag_up_right, axis=-1)))

    def get_valid_actions(self) -> list[int]:
        """Extracts valid actions from env.

        Returns:
            list of valid actions as integers (1-7).
        """
        # checks if top row is empty or not for each column
        playable = (self.board == -1).any(axis=0)
        return (np.nonzero(playable)[0] + 1).tolist()

    def decode_action(self, action: int | tuple[int, int]) -> tuple[int, int]:
        """Ensures action is encoded as tuple (row, col).

        Args:
            action: action as integer (1-7) or tuple (row, col).

        Returns:
            action as tuple (row, col).
        """
        if isinstance(action, (int | np.integer)):
            action_col = int(action) - 1  # convert to 0-6
            col_values = [row[action_col] for row in self.board]
            action_row = next((i for i, val in enumerate(col_values) if val == -1), None)
            if action_row is None:
                raise ValueError("Column already full.")
            action = (action_row, action_col)
        return action

    def render(self):
        """Prints current board for Connect Four."""
        current_sign = self.signs[self.current_player]
        print(f"-- Turn {self.turn_counter} | {current_sign}'s move --")

        # Print column numbers (1-7 for actions)
        print("  1   2   3   4   5   6   7 ")
        print(" ---------------------------")

        # Print 6 rows bottom-up (row 0 = bottom)
        for r in range(self.board.shape[0] - 1, -1, -1):  # Reverse to show top-first
            row_str = "|"
            for c in range(self.board.shape[1]):
                cell = self.board[r, c]
                sign = self.signs[cell] if cell != -1 else " "  # Empty as space
                row_str += f" {sign} |"
            print(row_str)
            print(" ---------------------------")

    def clone(self):
        """Returns a copy of the env."""
        new_env = object.__new__(Connect4Env)
        new_env.signs = self.signs
        new_env.rows = self.rows
        new_env.cols = self.cols
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.turn_counter = self.turn_counter
        new_env.done = self.done
        info = {"moves": self.info["moves"].copy()}
        if "winner" in self.info:
            info["winner"] = self.info["winner"]
        new_env.info = info
        return new_env
