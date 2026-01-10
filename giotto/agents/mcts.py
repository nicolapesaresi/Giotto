import numpy as np
from giotto.envs.generic import GenericEnv
from giotto.agents.algorithms.mcts import MCTS
from giotto.agents.generic import GenericAgent

class MCTSAgent(GenericAgent):
    """Selects action with MCTS."""
    def __init__(self, name: str="MCTSAgent", simulations:int=10000, cpuct:float=1.4):
        super().__init__(name)
        self.simulations = simulations
        self.cpuct = cpuct

    def select_action(self, env: GenericEnv) -> int:
        """Randomly selects valid action.
        Args:
            env: environment to extract valid actions from.
        Returns:
            action as integer (1-9 tris, 1-7 connect4).
        """
        mcts = MCTS(
            n_simulations=self.simulations,
            cpuct=self.cpuct
            )
        action = mcts.run(env)
        return action