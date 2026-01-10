from giotto.envs.generic import GenericEnv
from giotto.agents.generic import GenericAgent


class HumanAgent(GenericAgent):
    """Asks human input to move."""

    def __init__(self, name: str = "Human"):
        super().__init__(name)
        self.mode = "text"

    def select_action(self, env: GenericEnv) -> int:
        """Asks human for input and check action is valid.
        Args:
            env: environment to extract valid actions from.
        Returns:
            action as integer (1-9 tris, 1-7 connect4).
        """
        valid_actions = env.get_valid_actions()
        print(valid_actions)
        action: int = None
        while action not in valid_actions:
            if self.mode == "text":
                action = int(
                    input(f"Enter next move: [valid actions: {valid_actions}] ")
                )

        return action
