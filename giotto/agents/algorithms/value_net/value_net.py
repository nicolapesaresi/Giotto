import random
import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from giotto.envs.generic import GenericEnv
from giotto.agents.mcts import MCTSAgent


class ValueNet(nn.Module):
    def __init__(self, board_rows: int, board_cols, n_players: int = 2):
        super().__init__()
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.n_players = n_players

        # input shape: (n_players, board_rows, board_cols)
        self.conv1 = nn.Conv2d(n_players, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * self.board_rows * self.board_cols, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # value in [-1, 1]

        return x

    @staticmethod
    def process_state(state: list[np.ndarray, int]) -> torch.tensor:
        """Processes env state into value network input."""
        board = state[0]
        player_id = state[1]
        opp_id = 1 - player_id

        current = (board == player_id).astype(np.float32)
        opponent = (board == opp_id).astype(np.float32)

        input_array = np.stack([current, opponent], axis=0)
        return torch.from_numpy(input_array).unsqueeze(0)


class ValueNetTrainer:
    def __init__(self, env: GenericEnv):
        self.base_env = env
        self.net = ValueNet(
            board_cols=self.base_env.cols, board_rows=self.base_env.rows
        )

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def simulate_game(self, mcts_sims: int):
        new_memory = []
        states = []

        agents = [MCTSAgent(simulations=mcts_sims), MCTSAgent(simulations=mcts_sims)]
        env = self.base_env.clone()
        env.reset()

        while not env.done:
            state = env.get_state()
            states.append(state)
            action = agents[env.current_player].select_action(env)
            env.step(action)

        env_result = env.info["winner"]

        for state in states:
            if env_result == -1:
                result = 0
            else:
                result = 1 if env_result == state[1] else -1
            new_memory.append((self.net.process_state(state), result))
        return new_memory

    def generate_data(self, n_games: int, mcts_sims: int):
        memory = []
        for _ in tqdm(range(n_games), desc="Simulating games with MCTS..."):
            new_memory = self.simulate_game(mcts_sims)
            memory.extend(new_memory)
        return memory

    def train(self, epochs: int, memory: list):
        self.net.train()
        train_losses = []

        for epoch in range(epochs):
            random.shuffle(memory)
            total_loss = 0.0

            for state, result in memory:
                pred = self.net(state)

                target = torch.tensor([[result]], dtype=torch.float32)
                loss = F.mse_loss(pred, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(memory)
            train_losses.append(avg_loss)
            print(f"[Train] Epoch {epoch+1}/{epochs} - MSE: {avg_loss:.4f}")

        return train_losses

    @torch.no_grad()
    def test(self, memory: list):
        self.net.eval()
        total_loss = 0.0

        for state, result in memory:
            pred = self.net(state)
            target = torch.tensor([[result]], dtype=torch.float32)

            total_loss += F.mse_loss(pred, target).item()

        avg_loss = total_loss / len(memory)

        return avg_loss

    def save_metrics(
        self, path: str, train_losses: list, test_loss: float, config: dict
    ):
        results = {
            "train_mse": train_losses,
            "test_mse": test_loss,
            "config": config,
        }

        with open(path, "w") as f:
            json.dump(results, f)

    def save_model(self, path: str):
        torch.save(
            {
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def run(self, epochs: int, n_games: int, mcts_sims: int, log_dir: str):
        train_games = int(n_games * 0.8)
        test_games = n_games - train_games

        print("Generating train set...")
        train_memory = self.generate_data(train_games, mcts_sims)
        torch.save(train_memory, os.path.join(log_dir, "train_memory.pt"))

        train_losses = self.train(epochs, train_memory)

        print("Generating test set...")
        test_memory = self.generate_data(test_games, mcts_sims)
        torch.save(test_memory, os.path.join(log_dir, "test_memory.pt"))

        test_loss = self.test(test_memory)
        print(f"[Test] MSE: {test_loss:.4f}")

        self.save_model(os.path.join(log_dir, "valuenet.pt"))
        self.save_metrics(
            path=os.path.join(log_dir, "results_tris.json"),
            train_losses=train_losses,
            test_loss=test_loss,
            config={
                "game": args["game"],
                "epochs": epochs,
                "n_games": n_games,
                "mcts_sims": mcts_sims,
            },
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Value Net training")
    parser.add_argument(
        "-g", "--game", help="game to play [tris, connect4]", required=True
    )
    args = vars(parser.parse_args())

    # game
    if args["game"].lower() == "tris":
        from giotto.envs.tris import TrisEnv

        env = TrisEnv()
    elif args["game"].lower() == "connect4":
        from giotto.envs.connect4 import Connect4Env

        env = Connect4Env()
    else:
        raise ValueError(f"{args["game"]} not a valid game")

    LOGS_DIR = (
        Path(__file__).parent
        / ".."
        / ".."
        / ".."
        / ".."
        / "logs"
        / "value_net"
        / args["game"]
        / datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    )
    os.makedirs(LOGS_DIR)

    trainer = ValueNetTrainer(env)
    trainer.run(epochs=100, n_games=2000, mcts_sims=1000, log_dir=LOGS_DIR)
