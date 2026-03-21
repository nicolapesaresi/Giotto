import argparse

import numpy as np
import torch

from giotto.agents.algorithms.value_net.value_net import ValueNet


def convert_valuenet_pt_to_npz(pt_path: str, npz_path: str) -> None:
    """Convert a .pt checkpoint of a ValueNet to a .npz file.

    Args:
        pt_path: Path to the .pt checkpoint (saved as nested dict with "model_state_dict" key).
        npz_path: Destination path for the .npz file.
    """
    if "tris" in pt_path.lower():
        model = ValueNet(board_cols=3, board_rows=3)
    elif "connect4" in pt_path.lower():
        model = ValueNet(board_rows=6, board_cols=7)
    else:
        raise ValueError("Game not specified for ValueNet .pt path.")

    checkpoint = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])  # Nested key!
    model.eval()

    weights = {k: v.detach().cpu().numpy() for k, v in model.named_parameters()}
    np.savez_compressed(npz_path, **weights)
    print(f"Exported ValueNet .npz to {npz_path}")


def convert_alphazero_pt_to_npz(pt_path: str, npz_path: str) -> None:
    """Convert a .pt state-dict of an AlphaZeroNet to a .npz file.

    Saves the full state_dict (including BatchNorm running_mean / running_var)
    so the NumPy implementation can perform correct eval-mode inference.

    Args:
        pt_path: Path to the .pt file (saved directly as state_dict, not nested).
        npz_path: Destination path for the .npz file.
    """
    from giotto.agents.algorithms.alphazero.net import AlphaZeroNet

    if "tris" in pt_path.lower():
        model = AlphaZeroNet(
            input_size=[2, 3, 3], value_output_size=1, policy_output_size=9, channels=64, residual_blocks=5
        )
    elif "connect4" in pt_path.lower():
        model = AlphaZeroNet(
            input_size=[2, 6, 7], value_output_size=1, policy_output_size=7, channels=128, residual_blocks=10
        )
    else:
        raise ValueError("Game not specified for AlphaZeroNet .pt path.")

    # AlphaZero train.py saves state_dict directly (not nested).
    state_dict = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Use state_dict() (not named_parameters()) to capture BatchNorm buffers
    # (running_mean, running_var) needed for correct eval-mode inference.
    weights = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items() if "num_batches_tracked" not in k}
    np.savez_compressed(npz_path, **weights)
    print(f"Exported AlphaZeroNet .npz to {npz_path}")


def convert_pt_to_npz(pt_path: str, npz_path: str) -> None:
    """Auto-detect model type from filename and convert .pt to .npz.

    Args:
        pt_path: Path to the .pt file.
        npz_path: Destination path for the .npz file.
    """
    name = pt_path.lower()
    if "alphazero" in name:
        convert_alphazero_pt_to_npz(pt_path, npz_path)
    else:
        convert_valuenet_pt_to_npz(pt_path, npz_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .pt checkpoint to .npz")
    parser.add_argument("-p", "--path", help="path to .pt checkpoint", required=True)
    args = parser.parse_args()

    pt_path = args.path
    npz_path = pt_path.replace(".pt", ".npz")

    convert_pt_to_npz(pt_path, npz_path)
