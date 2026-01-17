import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    def __init__(self, board_rows: int, board_cols, n_players: int = 2):
        super().__init__()
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.n_players = n_players

        # input shape: (n_players, board_rows, board_cols)
        self.conv1 = nn.Conv2d(n_players, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * self.board_rows * self.board_cols, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # value in [-1, 1]

        return x

    @staticmethod
    def process_state(state: list[np.ndarray, int]) -> torch.tensor:
        """Processes env state into value network input."""
        board = state[0]
        player_id = state[1]
        opp_id = 1 - player_id

        current = (board == player_id).astype(np.float32)
        opponent = (board == opp_id).astype(np.float32)

        input_array = np.stack([current, opponent], axis=0)
        return torch.from_numpy(input_array).unsqueeze(0)
