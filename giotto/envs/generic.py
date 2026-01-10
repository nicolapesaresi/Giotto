import numpy as np


class GenericEnv:
    """Generic class for 2 players grid-based game."""

    def __init__(self, signs: list, rows: int, cols: int):
        """Instantiates environment."""
        self.signs = signs
        self.rows = rows
        self.cols = cols

    def reset(self, starting_player: int | None = None):
        """Resets environment to initial state.
        Args:
            starting_player: index of starting player (0=O, 1=X).
        """
        if starting_player not in (0, 1):
            starting_player = np.random.randint(2)
        self.current_player = starting_player
        self.board = np.full((self.rows, self.cols), fill_value=-1)
        self.turn_counter = 0

        self.done = False
        self.info = {}

    def step(self, action: int):
        """Updates environment after an action has been taken.
        Args:
            action: index of board cell that will be played by current player. Has to be integer (1-9 for tris, 1-7 for connect4).
        """
        assert (
            action in self.get_valid_actions()
        ), f"Received invalid action {action, type(action)}, has to be int in {self.get_valid_actions()}"

        # decode action
        action = self.decode_action(action)

        # play the move
        self.board[action] = self.current_player
        self.turn_counter += 1

        # check win/draw
        has_won = self.check_win(self.current_player)
        if has_won:
            self.done = True
            self.info["winner"] = self.current_player
        elif self.turn_counter == self.rows * self.cols:
            self.done = True
            self.info["winner"] = -1  # Draw
        else:
            self.current_player = (self.current_player + 1) % 2

    def get_state(self) -> list[np.ndarray, int]:
        """Returns state of the env."""
        return [self.board, self.current_player]

    # -------------
    # game specific methods, to be implemented in the child classes
    # -------------
    def check_win(self, player_idx: int) -> bool:
        """Checks if player has won. Implemented in child env for the specific game.
                Args:
            player_idx: index of the player to check for a win.
        Returns:
            True if player has won, False otherwise.
        """
        raise NotImplementedError

    def get_valid_actions(self) -> list[int]:
        """Extracts valid actions from env.
        Returns:
            list of valid actions as integers (1-9).
        """
        raise NotImplementedError

    def decode_action(self, action: int | tuple[int, int]) -> tuple[int, int]:
        """Ensures action is encoded as tuple (row, col).
        Args:
            action: action as integer (1-9 tris, 1-7 connect4) or tuple (row, col).
        Returns:
            action as tuple (row, col).
        """
        raise NotImplementedError

    def clone(self):
        """Returns a copy of the env."""
        raise NotImplementedError

    def render(self):
        """Text rendering of the current env state."""
        raise NotImplementedError
