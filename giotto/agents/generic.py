from giotto.envs.generic import GenericEnv


class GenericAgent:
    """Generic class for game agent."""

    def __init__(self, name: str):
        """Instantiates agent."""
        self.name = name

    def select_action(self, env: GenericEnv) -> int:
        """Selects an action to be played. To be implemented in child classes.
        Args:
            env: environment to extract valid actions from.
        Returns:
            action as integer (1-9 tris, 1-7 connect4).
        """
        raise NotImplementedError
