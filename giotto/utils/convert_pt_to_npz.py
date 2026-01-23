import torch
import numpy as np
import argparse
from giotto.agents.algorithms.value_net.value_net import ValueNet


def convert_pt_to_npz(pt_path: str, npz_path: str):
    if "tris" in pt_path.lower():
        model = ValueNet(board_cols=3, board_rows=3)
    elif "connect4" in pt_path.lower():
        model = ValueNet(board_rows=6, board_cols=7)
    else:
        raise ValueError("Game not specified for ValueNet .pt path.")

    # Load CHECKPOINT dict
    checkpoint = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])  # Nested key!
    model.eval()

    # Extract model weights only (ignore optimizer)
    weights = {k: v.detach().cpu().numpy() for k, v in model.named_parameters()}
    np.savez_compressed(npz_path, **weights)
    print(f"Exported .npz model to {npz_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .pt checkpoint to .npz")
    parser.add_argument("-p", "--path", help="path to .pt checkpoint", required=True)
    args = parser.parse_args()

    pt_path = args.path
    npz_path = pt_path.replace(".pt", ".npz")

    convert_pt_to_npz(pt_path, npz_path)
