from __future__ import annotations

from pathlib import Path

from giotto.agents.algorithms.alphazero.mcts import AlphaZeroMCTS
from giotto.agents.generic import GenericAgent

# try except block to make it work in browser mode
try:
    BROWSER_MODE = False
    import torch

    from giotto.agents.algorithms.alphazero.net import AlphaZeroNet
except ImportError:
    BROWSER_MODE = True
    from giotto.agents.algorithms.alphazero.net_numpy import AlphaZeroNetNumpy as AlphaZeroNet


class AlphaZeroAgent(GenericAgent):
    """Selects action with MCTS and value network."""

    def __init__(
        self,
        name: str = "AlphaZeroAgent",
        game: str | None = None,
        simulations: int = 800,
        cpuct: float = 1.4,
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

        self.simulations = simulations
        self.cpuct = cpuct

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
        mcts = AlphaZeroMCTS(
            net=self.net,
            n_simulations=self.simulations,
            cpuct=self.cpuct,
        )
        action, _root = mcts.run(env, temperature=0.0)
        return action
