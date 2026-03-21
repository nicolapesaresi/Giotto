"""Export AlphaZero PyTorch models to ONNX format for browser inference."""

import argparse
from pathlib import Path

import numpy as np
import torch

from giotto.agents.algorithms.alphazero.net import AlphaZeroNet

GAMES = {
    "tris": {
        "input_size": [2, 3, 3],
        "policy_output_size": 9,
        "residual_blocks": 5,
        "channels": 64,
    },
    "connect4": {
        "input_size": [2, 6, 7],
        "policy_output_size": 7,
        "residual_blocks": 10,
        "channels": 128,
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def export_model(game: str, output_dir: Path | None = None):
    """Export a single game model to ONNX.

    Args:
        game: game name ("tris" or "connect4").
        output_dir: directory to save the .onnx file. Defaults to web/models/.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "web" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GAMES[game]
    pt_path = PROJECT_ROOT / "giotto" / "agents" / "models" / game / "alphazero.pt"

    net = AlphaZeroNet(
        input_size=config["input_size"],
        value_output_size=1,
        policy_output_size=config["policy_output_size"],
        channels=config["channels"],
        residual_blocks=config["residual_blocks"],
    )
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=True)
    net.load_state_dict(checkpoint)
    net.eval()

    c, h, w = config["input_size"]
    dummy = torch.randn(1, c, h, w)

    onnx_path = output_dir / f"{game}.onnx"
    torch.onnx.export(
        net,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={"input": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=13,
    )
    print(f"Exported {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB)")

    # Validate: compare PyTorch vs ONNX outputs
    _validate(net, dummy, onnx_path)


def _validate(net: AlphaZeroNet, dummy: torch.Tensor, onnx_path: Path):
    """Compare PyTorch and ONNX outputs on the same input."""
    import onnxruntime as ort

    with torch.inference_mode():
        pt_policy, pt_value = net(dummy)
    pt_policy = pt_policy.numpy()
    pt_value = pt_value.numpy()

    session = ort.InferenceSession(str(onnx_path))
    onnx_policy, onnx_value = session.run(None, {"input": dummy.numpy()})

    policy_diff = np.max(np.abs(pt_policy - onnx_policy))
    value_diff = np.max(np.abs(pt_value - onnx_value))
    print(f"  Policy max diff: {policy_diff:.2e}")
    print(f"  Value max diff:  {value_diff:.2e}")
    assert policy_diff < 1e-5, f"Policy mismatch: {policy_diff}"
    assert value_diff < 1e-5, f"Value mismatch: {value_diff}"
    print("  Validation passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export AlphaZero models to ONNX")
    parser.add_argument("-g", "--game", choices=["tris", "connect4", "all"], default="all")
    args = parser.parse_args()

    games = list(GAMES.keys()) if args.game == "all" else [args.game]
    for g in games:
        export_model(g)
