import numpy as np


class ValueNetNumpy:
    def __init__(self, board_rows: int, board_cols: int, n_players: int = 2):
        """Identical signature to Torch version."""
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.n_players = n_players

    @staticmethod
    def process_state(state):
        board, player_id = state
        opp_id = 1 - player_id
        current = (board == player_id).astype(np.float32)
        opponent = (board == opp_id).astype(np.float32)
        return np.stack([current, opponent], axis=0)[None]  # [1,2,H,W]

    def forward(self, x):
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
        cout, cink, kh, kw = w.shape
        hout, wout = hin, win  # padding=1 preserves

        out = np.zeros((batch, cout, hout, wout), dtype=np.float32)

        for b_i in range(batch):
            for cout_i in range(cout):
                for i in range(hout):
                    for j in range(wout):
                        conv_sum = 0.0
                        for cin_i in range(cin):
                            for di in range(kh):
                                for dj in range(kw):
                                    ii = i + di - padding
                                    jj = j + dj - padding
                                    if 0 <= ii < hin and 0 <= jj < win:
                                        conv_sum += (
                                            x[b_i, cin_i, ii, jj]
                                            * w[cout_i, cin_i, di, dj]
                                        )
                        out[b_i, cout_i, i, j] = conv_sum + b[cout_i]

        return out

    def load_numpy_weights(self, path):
        data = np.load(path)
        print(data)
        self.conv1_w, self.conv1_b = data["conv1.weight"], data["conv1.bias"]
        self.conv2_w, self.conv2_b = data["conv2.weight"], data["conv2.bias"]
        self.fc1_w, self.fc1_b = data["fc1.weight"], data["fc1.bias"]
        self.fc2_w, self.fc2_b = data["fc2.weight"], data["fc2.bias"]

    def __call__(self, x: np.ndarray):
        """EXACT nn.Module behavior: model(x) → forward(x)"""
        return self.forward(x)

    def eval(self):
        pass  # NumPy always eval

    def load_state_dict(self, state_dict):
        raise RuntimeError("Use load_numpy_weights() for NumPyValueNet")
