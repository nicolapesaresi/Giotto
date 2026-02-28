import bitbully as bb

from giotto.agents.generic import GenericAgent
from giotto.envs.generic import GenericEnv
from giotto.utils.bitbully_utils import convert_board


class BitBullyAgent(GenericAgent):
    """BitBully solver for Connect4."""

    def __init__(self, name: str = "BitBully"):
        super().__init__(name)
        self.mode = "text"
        self.solver = bb.BitBully()

    def select_action(self, env: GenericEnv) -> int:
        """Queries BitBully solver for best action.

        Args:
            env: environment to extract valid actions from.

        Returns:
            action as integer (1-9 tris, 1-7 connect4).
        """
        valid_actions = env.get_valid_actions()
        bb_board = convert_board(env.info["moves"])
        action = self.solver.best_move(bb_board) + 1
        if action in valid_actions:
            return action
        else:
            raise ValueError("Invalid action returned by BitBully solver.")
