import numpy as np


class AlphaZeroNetNumpy:
    """NumPy drop-in for the Torch AlphaZeroNet (inference only).

    Matches:
    - __init__(input_size, value_output_size, policy_output_size, channels=64, residual_blocks=5)
    - forward(x) -> (policy_logits, value)
    - process_state(state) -> x with shape [1,2,H,W]
    - predict(state) -> (policy_probs, value_scalar)
    - eval(), load_state_dict(), parameters()
    """

    def __init__(
        self,
        input_size: list[int, int, int],
        value_output_size: int,
        policy_output_size: int,
        channels: int = 64,
        residual_blocks: int = 5,
    ):
        self.input_size = input_size
        self.value_output_size = value_output_size
        self.policy_output_size = policy_output_size
        self.channels = channels
        self.residual_blocks = residual_blocks

        # BatchNorm eps: PyTorch default is 1e-5 unless you changed it. [web:46]
        self._bn_eps = 1e-5

        # Storage for weights loaded via load_state_dict()
        self._sd = {}
        self._device = "cpu"  # compatibility only

    # -------------------------
    # Torch-like API helpers
    # -------------------------
    def parameters(self):
        """Torch compatibility: allows next(self.parameters()).device pattern."""

        class _P:
            def __init__(self, device):  # noqa: D401
                self.device = device

        yield _P(self._device)

    def to(self, device):
        """Torch compatibility; no-op for NumPy."""
        self._device = str(device)
        return self

    def eval(self):
        """NumPy always eval; kept for compatibility."""
        return self

    # -------------------------
    # Core forward
    # -------------------------
    def forward(self, x):
        """x: np.ndarray [B,C,H,W] float32. Returns:
        policy_logits: [B, policy_output_size]
        value: [B, value_output_size].
        """
        x = x.astype(np.float32, copy=False)

        # trunk: conv + bn + relu
        x = self._conv2d(x, self._sd["conv.weight"], bias=None, padding=1, stride=1)
        x = self._bn2d_eval(
            x,
            self._sd["bn.weight"],
            self._sd["bn.bias"],
            self._sd["bn.running_mean"],
            self._sd["bn.running_var"],
        )
        x = self._relu(x)

        # residual tower
        for i in range(self.residual_blocks):
            x = self._res_block(x, i)

        # policy head
        p = self._conv2d(
            x,
            self._sd["policy_conv.weight"],
            bias=self._sd["policy_conv.bias"],
            padding=0,
            stride=1,
        )
        p = self._bn2d_eval(
            p,
            self._sd["policy_bn.weight"],
            self._sd["policy_bn.bias"],
            self._sd["policy_bn.running_mean"],
            self._sd["policy_bn.running_var"],
        )
        p = self._relu(p)
        p = p.reshape(p.shape[0], -1)
        p = self._linear(p, self._sd["policy_fc.weight"], self._sd["policy_fc.bias"])

        # value head
        v = self._conv2d(
            x,
            self._sd["value_conv.weight"],
            bias=self._sd["value_conv.bias"],
            padding=0,
            stride=1,
        )
        v = self._bn2d_eval(
            v,
            self._sd["value_bn.weight"],
            self._sd["value_bn.bias"],
            self._sd["value_bn.running_mean"],
            self._sd["value_bn.running_var"],
        )
        v = self._relu(v)
        v = v.reshape(v.shape[0], -1)
        v = self._relu(self._linear(v, self._sd["value_fc1.weight"], self._sd["value_fc1.bias"]))
        v = np.tanh(self._linear(v, self._sd["value_fc2.weight"], self._sd["value_fc2.bias"]))

        return p.astype(np.float32), v.astype(np.float32)

    # -------------------------
    # Same helpers as Torch version
    # -------------------------
    @staticmethod
    def process_state(state: list[np.ndarray, int]):
        """Process state as returned to env to network input."""
        board, player_id = state
        player_id = int(player_id)
        opp_id = 1 - player_id

        current = (board == player_id).astype(np.float32)
        opponent = (board == opp_id).astype(np.float32)

        x = np.stack([current, opponent], axis=0)
        return x[None]  # [1,2,H,W] numpy (not torch)

    def predict(self, state: list[np.ndarray, int]):
        """Returns (policy_probs, value) as numpy, like your Torch version."""
        self.eval()
        x = self.process_state(state)
        policy_logits, value = self.forward(x)
        policy_probs = self._softmax(policy_logits, axis=1)
        return policy_probs.squeeze(), value.squeeze()

    # -------------------------
    # Weight loading (Torch-style)
    # -------------------------
    def load_state_dict(self, state_dict):
        """Accepts a Torch-like state_dict.

        You can pass:
        - actual torch tensors (will be converted via .detach().cpu().numpy())
        - numpy arrays already
        """

        def to_np(a):
            if hasattr(a, "detach"):
                a = a.detach()
            if hasattr(a, "cpu"):
                a = a.cpu()
            if hasattr(a, "numpy"):
                a = a.numpy()
            return np.asarray(a, dtype=np.float32)

        required = [
            "conv.weight",
            "bn.weight",
            "bn.bias",
            "bn.running_mean",
            "bn.running_var",
            "policy_conv.weight",
            "policy_conv.bias",
            "policy_bn.weight",
            "policy_bn.bias",
            "policy_bn.running_mean",
            "policy_bn.running_var",
            "policy_fc.weight",
            "policy_fc.bias",
            "value_conv.weight",
            "value_conv.bias",
            "value_bn.weight",
            "value_bn.bias",
            "value_bn.running_mean",
            "value_bn.running_var",
            "value_fc1.weight",
            "value_fc1.bias",
            "value_fc2.weight",
            "value_fc2.bias",
        ]
        for k in required:
            if k not in state_dict:
                raise KeyError(f"Missing key in state_dict: {k}")

        # Base weights
        for k in required:
            self._sd[k] = to_np(state_dict[k])

        # Residual blocks
        for i in range(self.residual_blocks):
            for kk in [
                f"res_blocks.{i}.conv1.weight",
                f"res_blocks.{i}.bn1.weight",
                f"res_blocks.{i}.bn1.bias",
                f"res_blocks.{i}.bn1.running_mean",
                f"res_blocks.{i}.bn1.running_var",
                f"res_blocks.{i}.conv2.weight",
                f"res_blocks.{i}.bn2.weight",
                f"res_blocks.{i}.bn2.bias",
                f"res_blocks.{i}.bn2.running_mean",
                f"res_blocks.{i}.bn2.running_var",
            ]:
                if kk not in state_dict:
                    raise KeyError(f"Missing key in state_dict: {kk}")
                self._sd[kk] = to_np(state_dict[kk])

        # Note: BatchNorm also has num_batches_tracked buffers in state_dict sometimes;
        # we ignore them safely. [web:66]
        return self

    # -------------------------
    # Internal ops
    # -------------------------
    def _res_block(self, x, i: int):
        residual = x

        y = self._conv2d(x, self._sd[f"res_blocks.{i}.conv1.weight"], bias=None, padding=1, stride=1)
        y = self._bn2d_eval(
            y,
            self._sd[f"res_blocks.{i}.bn1.weight"],
            self._sd[f"res_blocks.{i}.bn1.bias"],
            self._sd[f"res_blocks.{i}.bn1.running_mean"],
            self._sd[f"res_blocks.{i}.bn1.running_var"],
        )
        y = self._relu(y)

        y = self._conv2d(y, self._sd[f"res_blocks.{i}.conv2.weight"], bias=None, padding=1, stride=1)
        y = self._bn2d_eval(
            y,
            self._sd[f"res_blocks.{i}.bn2.weight"],
            self._sd[f"res_blocks.{i}.bn2.bias"],
            self._sd[f"res_blocks.{i}.bn2.running_mean"],
            self._sd[f"res_blocks.{i}.bn2.running_var"],
        )

        y = y + residual
        return self._relu(y)

    def _relu(self, x):
        return np.maximum(x, 0.0)

    def _softmax(self, x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=axis, keepdims=True)

    def _linear(self, x, w, b):
        # PyTorch Linear: y = x @ w.T + b, with w shape [out,in]. [web:76]
        return x @ w.T + b

    def _bn2d_eval(self, x, gamma, beta, running_mean, running_var):
        # Eval BN uses running stats. [web:46]
        mean = running_mean[None, :, None, None]
        var = running_var[None, :, None, None]
        gamma = gamma[None, :, None, None]
        beta = beta[None, :, None, None]
        return (x - mean) / np.sqrt(var + self._bn_eps) * gamma + beta

    def _conv2d(self, x, w, bias=None, padding=1, stride=1):
        """PyTorch-like Conv2d (actually cross-correlation). [web:52]."""
        batch, cin, hin, win = x.shape
        cout, cin_w, kh, kw = w.shape
        if cin != cin_w:
            raise ValueError(f"Conv cin mismatch: x has {cin}, w expects {cin_w}")

        hout = (hin + 2 * padding - kh) // stride + 1
        wout = (win + 2 * padding - kw) // stride + 1

        x_pad = np.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
        ).astype(np.float32, copy=False)

        # im2col
        cols = np.zeros((batch, cin, kh, kw, hout, wout), dtype=np.float32)
        for di in range(kh):
            for dj in range(kw):
                cols[:, :, di, dj, :, :] = x_pad[
                    :,
                    :,
                    di : di + stride * hout : stride,
                    dj : dj + stride * wout : stride,
                ]

        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(batch * hout * wout, -1)
        w_col = w.reshape(cout, -1).astype(np.float32, copy=False)

        out = cols @ w_col.T  # [B*hout*wout, cout]
        if bias is not None:
            out += bias.astype(np.float32, copy=False)[None, :]

        out = out.reshape(batch, hout, wout, cout).transpose(0, 3, 1, 2)
        return out.astype(np.float32, copy=False)
