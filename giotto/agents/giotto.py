from __future__ import annotations
import torch
from pathlib import Path
from giotto.agents.generic import GenericAgent
from giotto.agents.algorithms.mcts import MCTS
from giotto.agents.algorithms.value_net.value_net import ValueNet


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
            self.load_valuenet(
                Path(__file__).parent / "models" / game.lower() / "valuenet.pt"
            )

        self.simulations = simulations
        self.cpuct = cpuct

    def load_valuenet(self, path: str):
        """Loads value network model."""
        checkpoint = torch.load(path, map_location="cpu")
        self.valuenet.load_state_dict(checkpoint["model_state_dict"])
        self.valuenet.eval()

    def select_action(self, env):
        mcts = MCTS(
            n_simulations=self.simulations, cpuct=self.cpuct, valuenet=self.valuenet
        )
        action = mcts.run(env)
        return action

    # def process_state(self, state: list[np.ndarray, int]) -> torch.tensor:
    #     """Processes env state into value network input."""
    #     return self.valuenet.process_state(state)

    # def select_action(self, env: GenericEnv) -> int:
    #     """Randomly selects valid action.
    #     Args:
    #         env: environment to extract valid actions from.
    #     Returns:
    #         action as integer (1-9 tris, 1-7 connect4).
    #     """
    #     valid_actions = env.get_valid_actions()
    #     evaluations = []

    #     for candidate in valid_actions:
    #         child_env = env.clone()
    #         child_env.step(candidate)

    #         value_input = self.process_state(child_env.get_state())
    #         evaluations.append(self.valuenet(value_input).item())

    #     # minimize opponent's value
    #     best_value = min(evaluations)
    #     best_actions = [
    #         action
    #         for action, value in zip(valid_actions, evaluations)
    #         if value == best_value
    #     ]
    #     return random.choice(best_actions)
