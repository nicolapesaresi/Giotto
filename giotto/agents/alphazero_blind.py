from __future__ import annotations

from pathlib import Path

import numpy as np

from giotto.agents.generic import GenericAgent

# try except block to make it work in browser mode
try:
    BROWSER_MODE = False
    import torch

    from giotto.agents.algorithms.alphazero.net import AlphaZeroNet
except ImportError:
    BROWSER_MODE = True
    from giotto.agents.algorithms.alphazero.net_numpy import AlphaZeroNetNumpy as AlphaZeroNet


class BlindAlphaZeroAgent(GenericAgent):
    """Selects action with AlphaZero style policy-value network, without using MCTS to refine the prediction."""

    def __init__(
        self,
        name: str = "BlindAlphaZeroAgent",
        game: str | None = None,
        net: AlphaZeroNet | None = None,
    ):
        super().__init__(name)
        # load model
        if net:
            self.net = net
        else:
            # TODO: use model config json
            if game.lower() == "tris":
                self.net = AlphaZeroNet(
                    input_size=[2, 3, 3], value_output_size=1, policy_output_size=9, channels=64, residual_blocks=5
                )
            elif game.lower() == "connect4":
                self.net = AlphaZeroNet(
                    input_size=[2, 6, 7], value_output_size=1, policy_output_size=7, channels=128, residual_blocks=10
                )
            else:
                raise ValueError(f"Game {game} not supported.")
            if not BROWSER_MODE:
                self.load_net(Path(__file__).parent / "models" / game.lower() / "alphazero.pt")
            else:
                self.net.load_numpy_weights(Path(__file__).parent / "models" / game.lower() / "alphazero.npz")

    def load_net(self, path: str):
        """Loads value network model."""
        checkpoint = torch.load(path, map_location="cpu")
        self.net.load_state_dict(checkpoint)
        self.net.eval()

    def load_valuenet_numpy(self, path: str):
        """Loads value network model as numpy version for browser mode."""
        self.net.load_state_dict(path)

    def select_action(self, env):
        """Selects action using AlphaZero style network."""
        valid_actions = env.get_valid_actions()

        policy, value = self.net.predict(env.get_state())
        raw = np.array([policy[a - 1] for a in valid_actions], dtype=np.float32)
        raw /= raw.sum()

        best_idx = np.argmax(raw)
        action = valid_actions[best_idx]
        return action
