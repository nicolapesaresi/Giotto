import numpy as np


class ValueNetNumpy:
    """NumPy implementation of ValueNet for 2-player board games."""

    def __init__(self, board_rows: int, board_cols: int, n_players: int = 2):
        """Identical signature to Torch version."""
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.n_players = n_players

    @staticmethod
    def process_state(state):
        """Processes env state into value network input."""
        board, player_id = state
        opp_id = 1 - player_id
        current = (board == player_id).astype(np.float32)
        opponent = (board == opp_id).astype(np.float32)
        return np.stack([current, opponent], axis=0)[None]  # [1,2,H,W]

    def forward(self, x):
        """Forward pass of value network."""
        # Exact Conv2D → ReLU → Flatten → FC → tanh
        x = self._conv2d(x, self.conv1_w, self.conv1_b)
        x = np.maximum(x, 0)  # ReLU

        x = self._conv2d(x, self.conv2_w, self.conv2_b)
        x = np.maximum(x, 0)  # ReLU

        x = x.reshape(1, -1)
        x = np.maximum(np.matmul(x, self.fc1_w.T) + self.fc1_b, 0)  # ReLU
        x = np.tanh(np.matmul(x, self.fc2_w.T) + self.fc2_b)
        return x[0, 0]

    def _conv2d(self, x, w, b, padding=1, stride=1):
        """PyTorch nn.Conv2d replica."""
        batch, cin, hin, win = x.shape
        cout, cin_w, kh, kw = w.shape
        assert cin == cin_w

        # Output spatial size (padding=1, stride=1 preserves size)
        hout = (hin + 2 * padding - kh) // stride + 1
        wout = (win + 2 * padding - kw) // stride + 1

        # Pad input
        x_pad = np.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
        )

        # ---- im2col ----
        # Shape: (batch, cin, kh, kw, hout, wout)
        cols = np.zeros((batch, cin, kh, kw, hout, wout), dtype=x.dtype)

        for di in range(kh):
            for dj in range(kw):
                cols[:, :, di, dj, :, :] = x_pad[
                    :,
                    :,
                    di : di + stride * hout : stride,
                    dj : dj + stride * wout : stride,
                ]

        # (batch * hout * wout, cin * kh * kw)
        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(batch * hout * wout, -1)
        w_col = w.reshape(cout, -1)
        out = cols @ w_col.T  # (batch*hout*wout, cout)
        out += b
        out = out.reshape(batch, hout, wout, cout).transpose(0, 3, 1, 2)

        return out.astype(np.float32)

    def load_numpy_weights(self, path):
        """Loads weights from .npz file."""
        data = np.load(path)
        self.conv1_w, self.conv1_b = data["conv1.weight"], data["conv1.bias"]
        self.conv2_w, self.conv2_b = data["conv2.weight"], data["conv2.bias"]
        self.fc1_w, self.fc1_b = data["fc1.weight"], data["fc1.bias"]
        self.fc2_w, self.fc2_b = data["fc2.weight"], data["fc2.bias"]

    def __call__(self, x: np.ndarray):
        """Replica of torch nn.Module behavior: model(x) → forward(x)."""
        return self.forward(x)

    def eval(self):
        """Dummy eval() method for compatibility."""  # noqa: D402
        pass  # NumPy always eval

    def load_state_dict(self, state_dict):
        """Dummy method for compatibility."""
        raise RuntimeError("Use load_numpy_weights() for NumPyValueNet")
