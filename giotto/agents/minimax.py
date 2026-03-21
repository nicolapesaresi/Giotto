import math

from giotto.agents.generic import GenericAgent
from giotto.envs.generic import GenericEnv


class MinimaxAgent(GenericAgent):
    """Minimax-based TicTacToe agent."""

    def __init__(self, name="Minimax"):
        """Instantiates agent."""
        super().__init__(name)
        self.player_id = None

    def select_action(self, env: GenericEnv) -> int:
        """Selects action using minimax algorithm."""
        if self.player_id is None:
            self.player_id = env.current_player

        best_score = -math.inf
        best_action = None

        for action in env.get_valid_actions():
            sim_env = env.clone()
            sim_env.step(action)
            score = self._minimax(sim_env, depth=1)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _minimax(self, env, depth=0) -> int:
        """Minimax algorithm."""
        if env.done:
            return self._evaluate(env, depth)

        if env.current_player == self.player_id:
            best = -math.inf
            for action in env.get_valid_actions():
                sim_env = env.clone()
                sim_env.step(action)
                best = max(best, self._minimax(sim_env, depth + 1))
            return best
        else:
            best = math.inf
            for action in env.get_valid_actions():
                sim_env = env.clone()
                sim_env.step(action)
                best = min(best, self._minimax(sim_env, depth + 1))
            return best

    def _evaluate(self, env, depth) -> int:
        """Position evaluation."""
        winner = env.info["winner"]

        if winner == -1:
            return 0

        elif winner == self.player_id:
            return 10 - depth  # faster win = larger score

        else:
            return depth - 10  # slower loss = better than fast loss

    def evaluate_all_moves(self, env: GenericEnv) -> dict:
        """Evaluates all valid moves."""
        if self.player_id is None:
            self.player_id = env.current_player

        move_scores = {}

        for action in env.get_valid_actions():
            sim_env = env.clone()
            sim_env.step(action)
            score = self._minimax(sim_env)
            move_scores[action] = score

        return move_scores
