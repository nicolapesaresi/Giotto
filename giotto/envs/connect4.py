import numpy as np
from giotto.envs.generic import GenericEnv

class Connect4Env(GenericEnv):
    """Connect4 environment."""
    def __init__(self):
        """Instantiates environment."""
        signs = ["o", "x", -1] # third is empty place, accessed with -1
        rows = 6
        cols = 7
        super().__init__(signs, rows, cols)
        self.reset()

    def check_win(self, player_idx: int) -> bool:
        board = self.board
        # horizontal
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if np.all(board[r, c:c+4] == player_idx):
                    return True
        # vertical
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if np.all(board[r:r+4, c] == player_idx):
                    return True
        # diagonal (top-left → bottom-right)
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if all(board[r+i, c+i] == player_idx for i in range(4)):
                    return True
        # diagonal / (bottom-left → top-right)
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if all(board[r-i, c+i] == player_idx for i in range(4)):
                    return True
        return False

    def get_valid_actions(self) -> list[int]:
        """Extracts valid actions from env.
        Returns:
            list of valid actions as integers (1-7).
        """
        valid_actions = []
        for c in range(self.cols):
            col_values = [row[c] for row in self.board]
            if -1 in col_values:  # Has empty slot
                valid_actions.append(c + 1)  # 1-7
        return valid_actions

    def decode_action(self, action: int|tuple[int,int]) -> tuple[int,int]:
        """Ensures action is encoded as tuple (row, col).
        Args:
            action: action as integer (1-7) or tuple (row, col).
        Returns:
            action as tuple (row, col).
        """
        if isinstance(action, int):
            action_col = action - 1  # convert to 0-6
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
                sign = self.signs[cell] if cell != -1 else ' '  # Empty as space
                row_str += f" {sign} |"
            print(row_str)
            print(" ---------------------------")

    def clone(self):
        """Returns a copy of the env."""
        new_env = Connect4Env()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.turn_counter = self.turn_counter
        new_env.done = self.done
        new_env.info = self.info.copy()
        return new_env
