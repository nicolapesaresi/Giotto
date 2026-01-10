import numpy as np
from giotto.envs.generic import GenericEnv

class TrisEnv(GenericEnv):
    """Tic Tac Toe environment."""
    def __init__(self):
        """Instantiates environment."""
        signs = ["o", "x", " "] # third is empty place, accessed with -1
        rows = 3
        cols = 3
        super().__init__(signs, rows, cols)
        self.reset()

    def check_win(self, player_idx: int) -> bool:
        """Checks if player has won.
        Args:
            player_idx: index of the player to check for a win.
        Returns:
            True if player has won, False otherwise.
        """
        board = self.board
        assert self.rows == self.cols, "Different number of rows and columns. Not supported."
        for i in range(self.rows):
            if np.all(board[i, :] == player_idx) or np.all(board[:, i] == player_idx):
                return True
        if np.all(np.diag(board) == player_idx) or np.all(np.diag(np.fliplr(board)) == player_idx):
            return True
        return False

    def get_valid_actions(self) -> list[int]:
        """Extracts valid actions from env.
        Returns:
            list of valid actions as integers (1-9).
        """
        valid_actions = []
        for r in range(3):
            for c in range(3):
                if self.board[r, c] == -1:
                    action_int = r * 3 + c + 1  # convert to 1-9
                    valid_actions.append(action_int)
        return valid_actions

    def decode_action(self, action: int|tuple[int,int])-> tuple[int,int]:
        """Ensures action is encoded as tuple (row, col).
        Args:
            action: action as integer (1-9) or tuple (row, col).
        Returns:
            action as tuple (row, col).
        """
        if isinstance(action, int):
            action_int = action - 1  # convert to 0-8
            action = (action_int // 3, action_int % 3)
        return action

    def render(self):
        """Prints current board."""
        current_sign = self.signs[self.current_player]
        print(f"-- Turn {self.turn_counter} | {current_sign}'s move --")
        
        print(f" {self.signs[self.board[0,0]]} | {self.signs[self.board[0,1]]} | {self.signs[self.board[0,2]]} ")
        print("---|---|---")
        print(f" {self.signs[self.board[1,0]]} | {self.signs[self.board[1,1]]} | {self.signs[self.board[1,2]]} ")
        print("---|---|---")
        print(f" {self.signs[self.board[2,0]]} | {self.signs[self.board[2,1]]} | {self.signs[self.board[2,2]]} ")

    def clone(self):
        """Returns a copy of the env."""
        new_env = TrisEnv()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.turn_counter = self.turn_counter
        new_env.done = self.done
        new_env.info = self.info.copy()
        return new_env
    
