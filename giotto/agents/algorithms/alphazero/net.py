import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    """Residual block for AlphaZero network."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """Forward pass thourgh block."""
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class AlphaZeroNet(nn.Module):
    """AlphaZero-style network:
    Shared convolutional trunk with a Policy head and a Value head.
    """

    def __init__(
        self,
        input_size: list[int, int, int],
        value_output_size: int,
        policy_output_size: int,
        channels: int = 64,
        residual_blocks: int = 5,
    ):
        """Instantiates neural network.

        Args:
            input_size: input shape in format [n_players, rows, cols].
            value_output_size: output shape of value head, usually one number.
            policy_output_size: output shape of policy head, corresponding to number of possible actions.
            channels: feature planes in shared trunk residual blocks.
            residual_blocks: number of residual blocks in the trunk.
        """
        super().__init__()

        self.input_size = input_size
        self.value_output_size = value_output_size
        self.policy_output_size = policy_output_size
        self.channels = channels
        self.residual_blocks = residual_blocks

        # initial convolution
        self.conv = nn.Conv2d(self.input_size[0], self.channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.channels)

        # residual tower
        self.res_blocks = nn.ModuleList([ResidualBlock(self.channels) for _ in range(self.residual_blocks)])

        # ------------------
        # policy head
        # ------------------
        self.policy_conv = nn.Conv2d(self.channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.input_size[1] * self.input_size[2], self.policy_output_size)

        # ------------------
        # value head
        # ------------------
        self.value_conv = nn.Conv2d(self.channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.input_size[1] * self.input_size[2], 64)
        self.value_fc2 = nn.Linear(64, self.value_output_size)

    def forward(self, x):
        """Forward pass though network."""
        # shared trunk
        x = F.relu(self.bn(self.conv(x)))

        for block in self.res_blocks:
            x = block(x)

        # policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    @staticmethod
    def process_state(state: list[np.ndarray, int]):
        """Process state as returned to env to network input."""
        board, player_id = state
        opp_id = 1 - player_id

        current = (board == player_id).astype(np.float32)
        opponent = (board == opp_id).astype(np.float32)

        x = np.stack([current, opponent], axis=0)
        return torch.from_numpy(x).unsqueeze(0)

    def predict(self, state: list[np.ndarray, int]):
        """Predict (policy_probs, value) as numpy."""
        device = next(self.parameters()).device

        input_tensor = self.process_state(state).to(device)

        with torch.inference_mode():
            policy_logits, value_tensor = self.forward(input_tensor)
            policy_probs = F.softmax(policy_logits, dim=1)

        policy = policy_probs.detach().cpu().numpy().squeeze()
        value = value_tensor.detach().cpu().numpy().squeeze()
        return policy, value
