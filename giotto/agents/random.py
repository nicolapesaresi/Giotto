import numpy as np
from giotto.envs.generic import GenericEnv
from giotto.agents.generic import GenericAgent

class RandomAgent(GenericAgent):
    """Selects random valid action."""
    def __init__(self, name: str="RandomAgent"):
        super().__init__(name)

    def select_action(self, env: GenericEnv) -> int:
        """Randomly selects valid action.
        Args:
            env: environment to extract valid actions from.
        Returns:
            action as integer (1-9 tris, 1-7 connect4).
        """
        valid_actions = env.get_valid_actions()
        action = np.random.choice(valid_actions)
        return int(action)
        