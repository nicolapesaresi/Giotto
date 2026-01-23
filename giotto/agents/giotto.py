from __future__ import annotations
from pathlib import Path
from giotto.agents.generic import GenericAgent
from giotto.agents.algorithms.mcts import MCTS

# try except block to make it work in browser mode
try:
    BROWSER_MODE = False
    import torch
    from giotto.agents.algorithms.value_net.value_net import ValueNet
except ImportError:
    BROWSER_MODE = True
    from giotto.agents.algorithms.value_net.value_net_numpy import (
        ValueNetNumpy as ValueNet,
    )


class GiottoAgent(GenericAgent):
    """Selects action with MCTS and value network."""

    def __init__(
        self,
        name: str = "Giotto",
        game: str | None = None,
        simulations: int = 1000,
        cpuct: float = 1.4,
        valuenet: ValueNet | None = None,
    ):
        super().__init__(name)
        # load model
        if valuenet:
            self.valuenet = valuenet
        else:
            if game.lower() == "tris":
                self.valuenet = ValueNet(3, 3)
            elif game.lower() == "connect4":
                self.valuenet = ValueNet(6, 7)
            else:
                raise ValueError(f"Game {game} not supported.")
            if not BROWSER_MODE:
                self.load_valuenet(
                    Path(__file__).parent / "models" / game.lower() / "valuenet.pt"
                )
            else:
                self.load_valuenet_numpy(
                    Path(__file__).parent / "models" / game.lower() / "valuenet.npz"
                )

        self.simulations = simulations
        self.cpuct = cpuct

    def load_valuenet(self, path: str):
        """Loads value network model."""
        checkpoint = torch.load(path, map_location="cpu")
        self.valuenet.load_state_dict(checkpoint["model_state_dict"])
        self.valuenet.eval()

    def load_valuenet_numpy(self, path: str):
        """Loads value network model as numpy version for browser mode."""
        self.valuenet.load_numpy_weights(path)

    def select_action(self, env):
        mcts = MCTS(
            n_simulations=self.simulations, cpuct=self.cpuct, valuenet=self.valuenet
        )
        action = mcts.run(env)
        return action
