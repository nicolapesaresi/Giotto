"""Verify that AlphaZeroNetNumpy produces outputs numerically identical to AlphaZeroNet (PyTorch)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from giotto.agents.algorithms.alphazero.net import AlphaZeroNet
from giotto.agents.algorithms.alphazero.net_numpy import AlphaZeroNetNumpy


def _make_pair(input_size, policy_output_size, channels, residual_blocks):
    """Build a matched torch + numpy net pair with shared random weights."""
    torch_net = AlphaZeroNet(
        input_size=input_size,
        value_output_size=1,
        policy_output_size=policy_output_size,
        channels=channels,
        residual_blocks=residual_blocks,
    )
    torch_net.eval()

    numpy_net = AlphaZeroNetNumpy(
        input_size=input_size,
        value_output_size=1,
        policy_output_size=policy_output_size,
        channels=channels,
        residual_blocks=residual_blocks,
    )
    numpy_net.load_state_dict(torch_net.state_dict())
    return torch_net, numpy_net


# ─── shared fixtures ────────────────────────────────────────────────────────

TRIS_CFG = {"input_size": [2, 3, 3], "policy_output_size": 9, "channels": 8, "residual_blocks": 2}
C4_CFG = {"input_size": [2, 6, 7], "policy_output_size": 7, "channels": 8, "residual_blocks": 2}


@pytest.fixture(params=[TRIS_CFG, C4_CFG], ids=["tris", "connect4"])
def net_pair(request):
    cfg = request.param
    return _make_pair(**cfg), cfg


# ─── predict (single state) ──────────────────────────────────────────────────


class TestPredict:
    def test_policy_matches(self, net_pair):
        (torch_net, numpy_net), cfg = net_pair
        rows, cols = cfg["input_size"][1], cfg["input_size"][2]
        rng = np.random.default_rng(0)
        board = rng.integers(0, 2, size=(rows, cols))
        state = [board, 0]

        t_policy, _ = torch_net.predict(state)
        n_policy, _ = numpy_net.predict(state)

        np.testing.assert_allclose(t_policy, n_policy, rtol=1e-5, atol=1e-5)

    def test_value_matches(self, net_pair):
        (torch_net, numpy_net), cfg = net_pair
        rows, cols = cfg["input_size"][1], cfg["input_size"][2]
        rng = np.random.default_rng(1)
        board = rng.integers(0, 2, size=(rows, cols))
        state = [board, 1]

        _, t_value = torch_net.predict(state)
        _, n_value = numpy_net.predict(state)

        np.testing.assert_allclose(float(t_value), float(n_value), rtol=1e-5, atol=1e-5)

    def test_policy_sums_to_one(self, net_pair):
        (_, numpy_net), cfg = net_pair
        rows, cols = cfg["input_size"][1], cfg["input_size"][2]
        board = np.zeros((rows, cols), dtype=np.int64)
        policy, _ = numpy_net.predict([board, 0])
        assert abs(policy.sum() - 1.0) < 1e-6

    def test_value_in_range(self, net_pair):
        (_, numpy_net), cfg = net_pair
        rows, cols = cfg["input_size"][1], cfg["input_size"][2]
        board = np.zeros((rows, cols), dtype=np.int64)
        _, value = numpy_net.predict([board, 0])
        assert -1.0 <= float(value) <= 1.0


# ─── batch_predict ───────────────────────────────────────────────────────────


class TestBatchPredict:
    def test_batch_policy_matches(self, net_pair):
        (torch_net, numpy_net), cfg = net_pair
        rows, cols = cfg["input_size"][1], cfg["input_size"][2]
        rng = np.random.default_rng(2)

        states = [[rng.integers(0, 2, size=(rows, cols)), i % 2] for i in range(4)]

        t_policies, t_values = torch_net.batch_predict(states)
        n_policies, n_values = numpy_net.batch_predict(states)

        np.testing.assert_allclose(t_policies, n_policies, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(t_values, n_values, rtol=1e-5, atol=1e-5)

    def test_batch_single_consistent_with_predict(self, net_pair):
        """batch_predict on a single state must agree with predict."""
        (_, numpy_net), cfg = net_pair
        rows, cols = cfg["input_size"][1], cfg["input_size"][2]
        rng = np.random.default_rng(3)
        board = rng.integers(0, 2, size=(rows, cols))

        policy_single, value_single = numpy_net.predict([board, 0])
        policies_batch, values_batch = numpy_net.batch_predict([[board, 0]])

        np.testing.assert_allclose(policy_single, policies_batch[0], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(float(value_single), float(values_batch[0]), rtol=1e-6, atol=1e-6)


# ─── npz round-trip via converter ───────────────────────────────────────────


class TestNpzRoundTrip:
    def test_load_numpy_weights_matches_torch(self, net_pair, tmp_path):
        """Save torch weights to .npz, reload into numpy net, check outputs match."""
        (torch_net, _), cfg = net_pair
        rows, cols = cfg["input_size"][1], cfg["input_size"][2]

        npz_path = tmp_path / "alphazero.npz"

        # Save state_dict to npz (same logic as convert_alphazero_pt_to_npz)
        weights = {
            k: v.detach().cpu().numpy() for k, v in torch_net.state_dict().items() if "num_batches_tracked" not in k
        }
        np.savez_compressed(str(npz_path), **weights)

        numpy_net2 = AlphaZeroNetNumpy(
            input_size=cfg["input_size"],
            value_output_size=1,
            policy_output_size=cfg["policy_output_size"],
            channels=cfg["channels"],
            residual_blocks=cfg["residual_blocks"],
        )
        numpy_net2.load_numpy_weights(str(npz_path))

        rng = np.random.default_rng(4)
        board = rng.integers(0, 2, size=(rows, cols))
        state = [board, 0]

        t_policy, t_value = torch_net.predict(state)
        n_policy, n_value = numpy_net2.predict(state)

        np.testing.assert_allclose(t_policy, n_policy, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(float(t_value), float(n_value), rtol=1e-5, atol=1e-5)


# ─── converter integration test (requires actual model files) ─────────────────

TRIS_PT = Path(__file__).parent.parent / "giotto" / "agents" / "models" / "tris" / "alphazero.pt"
C4_PT = Path(__file__).parent.parent / "giotto" / "agents" / "models" / "connect4" / "alphazero.pt"


@pytest.mark.parametrize(
    "pt_path,cfg",
    [
        pytest.param(
            TRIS_PT, TRIS_CFG, id="tris", marks=pytest.mark.skipif(not TRIS_PT.exists(), reason="model file missing")
        ),
        pytest.param(
            C4_PT, C4_CFG, id="connect4", marks=pytest.mark.skipif(not C4_PT.exists(), reason="model file missing")
        ),
    ],
)
def test_converter_real_model(pt_path, cfg, tmp_path):
    """convert_alphazero_pt_to_npz output matches torch inference on the pre-trained model."""
    from giotto.utils.convert_pt_to_npz import convert_alphazero_pt_to_npz

    npz_path = str(tmp_path / "alphazero.npz")
    convert_alphazero_pt_to_npz(str(pt_path), npz_path)

    torch_net = AlphaZeroNet(
        input_size=cfg["input_size"],
        value_output_size=1,
        policy_output_size=cfg["policy_output_size"],
        channels=64 if "tris" in str(pt_path) else 128,
        residual_blocks=5 if "tris" in str(pt_path) else 10,
    )
    torch_net.load_state_dict(torch.load(str(pt_path), map_location="cpu"))
    torch_net.eval()

    numpy_net = AlphaZeroNetNumpy(
        input_size=cfg["input_size"],
        value_output_size=1,
        policy_output_size=cfg["policy_output_size"],
        channels=64 if "tris" in str(pt_path) else 128,
        residual_blocks=5 if "tris" in str(pt_path) else 10,
    )
    numpy_net.load_numpy_weights(npz_path)

    rows, cols = cfg["input_size"][1], cfg["input_size"][2]
    rng = np.random.default_rng(99)
    board = rng.integers(0, 2, size=(rows, cols))
    state = [board, 0]

    t_policy, t_value = torch_net.predict(state)
    n_policy, n_value = numpy_net.predict(state)

    np.testing.assert_allclose(t_policy, n_policy, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(float(t_value), float(n_value), rtol=1e-5, atol=1e-5)
