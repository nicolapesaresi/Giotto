"""Tests for TrisEnv and Connect4Env game logic."""

import numpy as np
import pytest

from giotto.envs.connect4 import Connect4Env
from giotto.envs.tris import TrisEnv


class TestTrisEnvReset:
    def test_board_all_empty(self):
        env = TrisEnv()
        env.reset(0)
        assert np.all(env.board == -1)

    def test_board_shape(self):
        env = TrisEnv()
        env.reset(0)
        assert env.board.shape == (3, 3)

    def test_turn_counter_zero(self):
        env = TrisEnv()
        env.reset(0)
        assert env.turn_counter == 0

    def test_not_done(self):
        env = TrisEnv()
        env.reset(0)
        assert not env.done

    def test_starting_player_respected(self):
        env = TrisEnv()
        env.reset(1)
        assert env.current_player == 1

    def test_moves_list_empty(self):
        env = TrisEnv()
        env.reset(0)
        assert env.info["moves"] == []

    def test_no_winner_key(self):
        env = TrisEnv()
        env.reset(0)
        assert "winner" not in env.info


class TestTrisEnvActions:
    def test_nine_actions_on_empty_board(self):
        env = TrisEnv()
        env.reset(0)
        assert env.get_valid_actions() == list(range(1, 10))

    def test_actions_decrease_after_step(self):
        env = TrisEnv()
        env.reset(0)
        env.step(5)
        valid = env.get_valid_actions()
        assert 5 not in valid
        assert len(valid) == 8

    def test_decode_action_int_corners(self):
        env = TrisEnv()
        env.reset(0)
        assert env.decode_action(1) == (0, 0)
        assert env.decode_action(3) == (0, 2)
        assert env.decode_action(7) == (2, 0)
        assert env.decode_action(9) == (2, 2)

    def test_decode_action_int_center(self):
        env = TrisEnv()
        env.reset(0)
        assert env.decode_action(5) == (1, 1)

    def test_decode_action_tuple_passthrough(self):
        env = TrisEnv()
        env.reset(0)
        assert env.decode_action((2, 1)) == (2, 1)


class TestTrisEnvStep:
    def test_places_piece_on_board(self):
        env = TrisEnv()
        env.reset(0)
        env.step(1)
        assert env.board[0, 0] == 0

    def test_alternates_player(self):
        env = TrisEnv()
        env.reset(0)
        env.step(1)
        assert env.current_player == 1
        env.step(2)
        assert env.current_player == 0

    def test_increments_turn_counter(self):
        env = TrisEnv()
        env.reset(0)
        env.step(1)
        env.step(2)
        assert env.turn_counter == 2

    def test_records_move_in_info(self):
        env = TrisEnv()
        env.reset(0)
        env.step(5)
        env.step(1)
        assert env.info["moves"] == [5, 1]


class TestTrisEnvWin:
    def test_win_top_row(self):
        # P0: 1(0,0),2(0,1),3(0,2)  P1: 4,5
        env = TrisEnv()
        env.reset(0)
        for action in [1, 4, 2, 5, 3]:
            if not env.done:
                env.step(action)
        assert env.done
        assert env.info["winner"] == 0

    def test_win_column(self):
        # P0 fills col 0: actions 1(0,0),4(1,0),7(2,0)  P1: 2,5
        env = TrisEnv()
        env.reset(0)
        for action in [1, 2, 4, 5, 7]:
            if not env.done:
                env.step(action)
        assert env.done
        assert env.info["winner"] == 0

    def test_win_main_diagonal(self):
        # P0: 1(0,0),5(1,1),9(2,2)  P1: 2,3
        env = TrisEnv()
        env.reset(0)
        for action in [1, 2, 5, 3, 9]:
            if not env.done:
                env.step(action)
        assert env.done
        assert env.info["winner"] == 0

    def test_win_anti_diagonal(self):
        # P0: 3(0,2),5(1,1),7(2,0)  P1: 1,2
        env = TrisEnv()
        env.reset(0)
        for action in [3, 1, 5, 2, 7]:
            if not env.done:
                env.step(action)
        assert env.done
        assert env.info["winner"] == 0

    def test_draw(self):
        # Full board draw: P0 at (0,0),(0,2),(1,0),(2,2),(2,1)  P1 at (0,1),(1,1),(1,2),(2,0)
        env = TrisEnv()
        env.reset(0)
        for action in [1, 2, 3, 5, 4, 6, 9, 7, 8]:
            env.step(action)
        assert env.done
        assert env.info["winner"] == -1

    def test_not_done_mid_game(self):
        env = TrisEnv()
        env.reset(0)
        env.step(1)
        assert not env.done
        assert "winner" not in env.info

    def test_player_still_alternates_after_terminal(self):
        # current_player must flip even on terminal move (MCTS relies on this)
        env = TrisEnv()
        env.reset(0)
        for action in [1, 4, 2, 5, 3]:
            if not env.done:
                env.step(action)
        assert env.current_player == 1  # flipped after P0's winning move

    def test_winning_cells_row(self):
        env = TrisEnv()
        env.reset(0)
        for action in [1, 4, 2, 5, 3]:
            if not env.done:
                env.step(action)
        cells = env.get_winning_cells(0)
        assert set(cells) == {(0, 0), (0, 1), (0, 2)}

    def test_winning_cells_diagonal(self):
        env = TrisEnv()
        env.reset(0)
        for action in [1, 2, 5, 3, 9]:
            if not env.done:
                env.step(action)
        cells = env.get_winning_cells(0)
        assert set(cells) == {(0, 0), (1, 1), (2, 2)}

    def test_winning_cells_none_when_no_win(self):
        env = TrisEnv()
        env.reset(0)
        env.step(1)
        assert env.get_winning_cells(0) is None


class TestTrisEnvClone:
    def test_clone_board_is_independent(self):
        env = TrisEnv()
        env.reset(0)
        env.step(5)
        clone = env.clone()
        clone.step(1)
        assert env.board[0, 0] == -1  # original untouched

    def test_clone_preserves_board(self):
        env = TrisEnv()
        env.reset(0)
        env.step(5)
        clone = env.clone()
        assert np.array_equal(clone.board, env.board)

    def test_clone_preserves_metadata(self):
        env = TrisEnv()
        env.reset(0)
        env.step(5)
        clone = env.clone()
        assert clone.current_player == env.current_player
        assert clone.turn_counter == env.turn_counter
        assert clone.info["moves"] == env.info["moves"]

    def test_clone_moves_list_is_independent(self):
        env = TrisEnv()
        env.reset(0)
        env.step(5)
        clone = env.clone()
        clone.step(1)
        assert env.info["moves"] == [5]

    def test_clone_preserves_winner(self):
        env = TrisEnv()
        env.reset(0)
        for action in [1, 4, 2, 5, 3]:
            if not env.done:
                env.step(action)
        clone = env.clone()
        assert clone.info["winner"] == env.info["winner"]


class TestConnect4EnvReset:
    def test_board_all_empty(self):
        env = Connect4Env()
        env.reset(0)
        assert np.all(env.board == -1)

    def test_board_shape(self):
        env = Connect4Env()
        env.reset(0)
        assert env.board.shape == (6, 7)

    def test_turn_counter_zero(self):
        env = Connect4Env()
        env.reset(0)
        assert env.turn_counter == 0

    def test_starting_player_respected(self):
        env = Connect4Env()
        env.reset(1)
        assert env.current_player == 1


class TestConnect4EnvActions:
    def test_seven_actions_on_empty_board(self):
        env = Connect4Env()
        env.reset(0)
        assert env.get_valid_actions() == list(range(1, 8))

    def test_gravity_places_at_bottom_row(self):
        # Row 0 is the bottom; first piece in any column lands at row 0
        env = Connect4Env()
        env.reset(0)
        env.step(1)
        assert env.board[0, 0] == 0

    def test_gravity_stacks_pieces(self):
        # Alternating plays in the same column stack correctly
        env = Connect4Env()
        env.reset(0)
        env.step(1)  # P0 → row 0
        env.step(1)  # P1 → row 1
        assert env.board[0, 0] == 0
        assert env.board[1, 0] == 1

    def test_full_column_excluded_from_valid_actions(self):
        # Fill col 1 with alternating players (3P0 at rows 0,2,4 and 3P1 at rows 1,3,5)
        # → no 4-in-a-row, col full
        env = Connect4Env()
        env.reset(0)
        for _ in range(6):
            env.step(1)
        assert 1 not in env.get_valid_actions()

    def test_full_column_raises_value_error(self):
        env = Connect4Env()
        env.reset(0)
        for _ in range(6):
            env.step(1)
        with pytest.raises(ValueError, match="Column already full"):
            env.step(1)


class TestConnect4EnvWin:
    def test_win_horizontal(self):
        # P0 fills bottom row cols 1-4; P1 uses cols 5-7
        env = Connect4Env()
        env.reset(0)
        for action in [1, 5, 2, 6, 3, 7, 4]:
            if not env.done:
                env.step(action)
        assert env.done
        assert env.info["winner"] == 0

    def test_win_vertical(self):
        # P0 stacks col 1 four times; P1 uses col 2
        env = Connect4Env()
        env.reset(0)
        for action in [1, 2, 1, 2, 1, 2, 1]:
            if not env.done:
                env.step(action)
        assert env.done
        assert env.info["winner"] == 0

    def test_win_diagonal_down_right(self):
        # P0 builds diagonal (0,0),(1,1),(2,2),(3,3) using neutral col 7 as filler
        # Sequence fills col2/col3/col4 bottom rows for P1 so P0 can land at higher rows
        env = Connect4Env()
        env.reset(0)
        for action in [1, 2, 2, 3, 7, 3, 3, 4, 7, 4, 7, 4, 4]:
            if not env.done:
                env.step(action)
        assert env.done
        assert env.info["winner"] == 0

    def test_win_diagonal_up_right(self):
        # P0 builds anti-diagonal (0,3),(1,2),(2,1),(3,0) using neutral col 7 as filler
        env = Connect4Env()
        env.reset(0)
        for action in [4, 3, 3, 2, 7, 2, 2, 1, 7, 1, 7, 1, 1]:
            if not env.done:
                env.step(action)
        assert env.done
        assert env.info["winner"] == 0

    def test_winning_cells_horizontal(self):
        env = Connect4Env()
        env.reset(0)
        for action in [1, 5, 2, 6, 3, 7, 4]:
            if not env.done:
                env.step(action)
        cells = env.get_winning_cells(0)
        assert cells is not None
        assert len(cells) == 4
        assert all(r == 0 for r, _ in cells)

    def test_winning_cells_vertical(self):
        env = Connect4Env()
        env.reset(0)
        for action in [1, 2, 1, 2, 1, 2, 1]:
            if not env.done:
                env.step(action)
        cells = env.get_winning_cells(0)
        assert cells is not None
        assert len(cells) == 4
        assert all(c == 0 for _, c in cells)

    def test_no_winning_cells_when_no_win(self):
        env = Connect4Env()
        env.reset(0)
        env.step(1)
        assert env.get_winning_cells(0) is None


class TestConnect4EnvClone:
    def test_clone_board_is_independent(self):
        env = Connect4Env()
        env.reset(0)
        env.step(1)
        clone = env.clone()
        clone.step(2)
        assert env.board[0, 1] == -1

    def test_clone_preserves_board(self):
        env = Connect4Env()
        env.reset(0)
        env.step(1)
        clone = env.clone()
        assert np.array_equal(clone.board, env.board)

    def test_clone_preserves_metadata(self):
        env = Connect4Env()
        env.reset(0)
        env.step(1)
        clone = env.clone()
        assert clone.current_player == env.current_player
        assert clone.turn_counter == env.turn_counter
        assert clone.info["moves"] == env.info["moves"]

    def test_clone_preserves_winner(self):
        env = Connect4Env()
        env.reset(0)
        for action in [1, 2, 1, 2, 1, 2, 1]:
            if not env.done:
                env.step(action)
        clone = env.clone()
        assert clone.info["winner"] == env.info["winner"]
