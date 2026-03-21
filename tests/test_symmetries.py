"""Tests for EquivalentBoards symmetry transformations."""

import numpy as np
import pytest

from giotto.envs.connect4 import Connect4Env
from giotto.envs.tris import TrisEnv
from giotto.utils.simmetries import (
    EquivalentBoards,
    identity,
    reflect_diag_ne_sw,
    reflect_diag_nw_se,
    reflect_horizontal,
    reflect_vertical,
    rotate90,
    rotate180,
    rotate270,
)


class TestTransformFunctions:
    def test_identity_returns_copy(self):
        board = np.array([[1, 2], [3, 4]])
        result = identity(board)
        assert np.array_equal(result, board)
        result[0, 0] = 99
        assert board[0, 0] == 1  # original unchanged

    def test_rotate90_clockwise(self):
        board = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = rotate90(board)
        # Clockwise 90°: top-left becomes top-right
        assert result[0, 2] == 1
        assert result[2, 2] == 3

    def test_rotate180(self):
        board = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = rotate180(board)
        assert result[0, 0] == 9
        assert result[2, 2] == 1

    def test_rotate270_inverse_of_90(self):
        board = np.arange(9).reshape(3, 3)
        assert np.array_equal(rotate270(rotate90(board)), board)

    def test_reflect_horizontal_flips_rows(self):
        board = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = reflect_horizontal(board)
        assert np.array_equal(result[0], [7, 8, 9])
        assert np.array_equal(result[2], [1, 2, 3])

    def test_reflect_vertical_flips_cols(self):
        board = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = reflect_vertical(board)
        assert np.array_equal(result[0], [3, 2, 1])

    def test_reflect_diag_nw_se_is_transpose(self):
        board = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert np.array_equal(reflect_diag_nw_se(board), board.T)

    def test_reflect_diag_ne_sw_involution(self):
        board = np.arange(9).reshape(3, 3)
        assert np.array_equal(reflect_diag_ne_sw(reflect_diag_ne_sw(board)), board)


class TestEquivalentBoardsTris:
    def test_asymmetric_board_gives_eight_variants(self):
        # All-unique values guarantee all 8 dihedral transforms are distinct
        board = np.arange(9).reshape(3, 3)
        result = TrisEnv.simmetries.get_equivalent_boards(board)
        assert len(result) == 8

    def test_symmetric_board_deduplication(self):
        # Empty board is invariant under all transforms → only 1 result
        board = np.full((3, 3), -1)
        result = TrisEnv.simmetries.get_equivalent_boards(board)
        assert len(result) == 1

    def test_results_are_numpy_arrays(self):
        board = np.zeros((3, 3), dtype=int)
        result = TrisEnv.simmetries.get_equivalent_boards(board)
        for r in result:
            assert isinstance(r, np.ndarray)

    def test_identity_always_first(self):
        board = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
        result = TrisEnv.simmetries.get_equivalent_boards(board)
        assert np.array_equal(result[0], board)


class TestEquivalentBoardsConnect4:
    def test_asymmetric_board_gives_two_variants(self):
        # Connect4 only has horizontal-flip symmetry, so 2 variants for an asymmetric board
        board = np.full((6, 7), -1)
        board[0, 0] = 0  # single piece in col 0 — not symmetric
        result = Connect4Env.simmetries.get_equivalent_boards(board)
        assert len(result) == 2

    def test_symmetric_board_deduplication(self):
        # A left-right symmetric board yields only 1 result
        board = np.full((6, 7), -1)
        board[0, 3] = 0  # center column piece is symmetric
        result = Connect4Env.simmetries.get_equivalent_boards(board)
        assert len(result) == 1

    def test_flip_is_correct(self):
        board = np.full((6, 7), -1)
        board[0, 0] = 0
        result = Connect4Env.simmetries.get_equivalent_boards(board)
        flipped = result[1]
        assert flipped[0, 6] == 0  # piece mirrored to col 6
        assert flipped[0, 0] == -1


class TestPolicyAugmentationTris:
    def test_policy_and_board_augmented_together(self):
        # All-unique values guarantee all 8 dihedral transforms are distinct
        board = np.arange(9).reshape(3, 3)
        policy = np.arange(9, dtype=np.float32)
        result = TrisEnv.simmetries.get_equivalent_boards(board, policy_targets=policy)
        assert len(result) == 8
        for aug_board, aug_policy in result:
            assert isinstance(aug_board, np.ndarray)
            assert aug_policy.shape == (9,)

    def test_identity_policy_unchanged(self):
        board = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
        policy = np.arange(9, dtype=np.float32)
        result = TrisEnv.simmetries.get_equivalent_boards(board, policy_targets=policy)
        aug_board, aug_policy = result[0]
        assert np.array_equal(aug_board, board)
        np.testing.assert_array_equal(aug_policy, policy)

    def test_rotate180_policy_reversed(self):
        # 180° rotation maps cell i to cell (8-i) on a flat 3×3 grid
        board = np.arange(9).reshape(3, 3)
        policy = np.arange(9, dtype=np.float32)
        eb = EquivalentBoards(
            rotate90=False,
            rotate180=True,
            rotate270=False,
            reflect_horizontal=False,
            reflect_vertical=False,
            reflect_diag_nw_se=False,
            reflect_diag_ne_sw=False,
        )
        result = eb.get_equivalent_boards(board, policy_targets=policy)
        # Result contains identity and 180° rotation
        _, aug_policy = result[1]
        expected = policy.reshape(3, 3)[::-1, ::-1].reshape(9)
        np.testing.assert_array_equal(aug_policy, expected)


class TestPolicyAugmentationConnect4:
    def test_reflect_vertical_reverses_column_policy(self):
        board = np.full((6, 7), -1)
        board[0, 0] = 0  # asymmetric board
        policy = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        result = Connect4Env.simmetries.get_equivalent_boards(board, policy_targets=policy)
        assert len(result) == 2
        _, aug_policy = result[1]
        np.testing.assert_array_equal(aug_policy, policy[::-1])

    def test_unsupported_transform_raises(self):
        eb = EquivalentBoards(
            rotate90=True,
            rotate180=False,
            rotate270=False,
            reflect_horizontal=False,
            reflect_vertical=False,
            reflect_diag_nw_se=False,
            reflect_diag_ne_sw=False,
        )
        board = np.arange(42).reshape(6, 7)
        policy = np.ones(7, dtype=np.float32)
        with pytest.raises(ValueError, match="not supported"):
            eb.get_equivalent_boards(board, policy_targets=policy)
