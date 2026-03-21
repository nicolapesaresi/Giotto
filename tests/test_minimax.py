"""Tests for MinimaxAgent correctness on TrisEnv."""

from giotto.agents.minimax import MinimaxAgent
from giotto.envs.tris import TrisEnv


class TestMinimaxTakesWin:
    def test_takes_immediate_diagonal_win(self):
        # P0: (0,0)=1, (1,1)=5  →  action 9=(2,2) completes main diagonal
        env = TrisEnv()
        env.reset(0)
        for action in [1, 2, 5, 4]:
            env.step(action)
        agent = MinimaxAgent()
        assert agent.select_action(env) == 9

    def test_takes_immediate_row_win(self):
        # P0: (0,0)=1, (0,1)=2  →  action 3=(0,2) completes top row
        env = TrisEnv()
        env.reset(0)
        for action in [1, 7, 2, 8]:
            env.step(action)
        agent = MinimaxAgent()
        assert agent.select_action(env) == 3


class TestMinimaxBlocksOpponent:
    def test_blocks_row_win(self):
        # P1 has (0,0)=action1 and (0,1)=action2; P0 must block at action 3
        # Setup: P0 plays 5,9  P1 plays 1,2
        env = TrisEnv()
        env.reset(0)
        for action in [5, 1, 9, 2]:
            env.step(action)
        agent = MinimaxAgent()
        assert agent.select_action(env) == 3

    def test_prefers_win_over_block(self):
        # P0 can win immediately AND must block — should take the win
        # P0: (0,0)=1, (0,1)=2 → action 3 wins for P0 (row)
        # P1: (1,0)=4, (1,1)=5 → action 6 would win for P1 (row)
        # P0 must take 3 (win) rather than 6 (block)
        env = TrisEnv()
        env.reset(0)
        for action in [1, 4, 2, 5]:
            env.step(action)
        agent = MinimaxAgent()
        assert agent.select_action(env) == 3


class TestMinimaxEvaluateAllMoves:
    def test_covers_all_valid_actions(self):
        env = TrisEnv()
        env.reset(0)
        env.step(1)
        env.step(9)
        agent = MinimaxAgent()
        scores = agent.evaluate_all_moves(env)
        assert set(scores.keys()) == set(env.get_valid_actions())

    def test_winning_move_outscores_losing_move(self):
        # P0: (0,0)=1, (0,1)=2 → action 3 wins; action 6 lets P1 win
        env = TrisEnv()
        env.reset(0)
        for action in [1, 4, 2, 5]:
            env.step(action)
        agent = MinimaxAgent()
        scores = agent.evaluate_all_moves(env)
        assert scores[3] > scores[6]

    def test_draw_scores_zero(self):
        # On a nearly-full board where all remaining moves lead to draws
        # P0: 1,3,6,7  P1: 2,4,5,9  — one cell left: 8 → draw
        env = TrisEnv()
        env.reset(0)
        for action in [1, 2, 3, 4, 6, 5, 7, 9]:
            env.step(action)
        assert env.get_valid_actions() == [8]
        agent = MinimaxAgent()
        scores = agent.evaluate_all_moves(env)
        assert scores[8] == 0


class TestMinimaxScoring:
    def test_depth_penalises_slow_win(self):
        # A win at depth 1 should score higher than a win at depth 3
        env = TrisEnv()
        env.reset(0)
        agent = MinimaxAgent()
        agent.player_id = 0

        # Simulate terminal envs at different depths
        env_win_fast = TrisEnv()
        env_win_fast.reset(0)
        for action in [1, 4, 2, 5, 3]:  # P0 wins top row
            if not env_win_fast.done:
                env_win_fast.step(action)

        env_win_slow = env_win_fast.clone()
        score_fast = agent._evaluate(env_win_fast, depth=1)
        score_slow = agent._evaluate(env_win_slow, depth=5)
        assert score_fast > score_slow

    def test_draw_scores_zero_regardless_of_depth(self):
        env = TrisEnv()
        env.reset(0)
        for action in [1, 2, 3, 5, 4, 6, 9, 7, 8]:
            env.step(action)
        agent = MinimaxAgent()
        agent.player_id = 0
        assert agent._evaluate(env, depth=1) == 0
        assert agent._evaluate(env, depth=9) == 0

    def test_loss_returns_negative_score(self):
        env = TrisEnv()
        env.reset(0)
        for action in [1, 4, 2, 5, 3]:  # P0 wins
            if not env.done:
                env.step(action)
        agent = MinimaxAgent()
        agent.player_id = 1  # agent is P1, so this is a loss
        assert agent._evaluate(env, depth=1) < 0
