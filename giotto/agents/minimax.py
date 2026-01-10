import math
from giotto.envs.generic import GenericEnv
from giotto.agents.generic import GenericAgent


class MinimaxAgent(GenericAgent):
    """Minimax-based TicTacToe agent."""

    def __init__(self, name="Minimax"):
        super().__init__(name)
        self.player_id = None

    def select_action(self, env: GenericEnv) -> int:
        if self.player_id is None:
            self.player_id = env.current_player

        best_score = -math.inf
        best_action = None

        for action in env.get_valid_actions():
            sim_env = env.clone()
            sim_env.step(action)
            score = self._minimax(sim_env)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _minimax(self, env) -> int:
        if env.done:
            return self._evaluate(env)

        if env.current_player == self.player_id:
            # Agent’s turn → maximize
            best = -math.inf
            for action in env.get_valid_actions():
                sim_env = env.clone()
                sim_env.step(action)
                best = max(best, self._minimax(sim_env))
            return best
        else:
            # Opponent’s turn → minimize
            best = math.inf
            for action in env.get_valid_actions():
                sim_env = env.clone()
                sim_env.step(action)
                best = min(best, self._minimax(sim_env))
            return best

    def _evaluate(self, env) -> int:
        winner = env.info["winner"]
        if winner == -1:
            return 0  # draw
        elif winner == self.player_id:
            return 1  # win
        else:
            return -1  # loss
