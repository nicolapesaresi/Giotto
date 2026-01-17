import random
import os
import json
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from giotto.agents.algorithms.value_net.value_net import ValueNet
from giotto.envs.generic import GenericEnv
from giotto.agents.generic import GenericAgent
from giotto.agents.random import RandomAgent
from giotto.agents.giotto import GiottoAgent


class ValueNetTrainer:
    def __init__(self, env: GenericEnv):
        self.base_env = env
        self.net = ValueNet(
            board_cols=self.base_env.cols, board_rows=self.base_env.rows
        )

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def simulate_game(self, agents: list):
        new_memory = []
        states = []

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

    def self_play(self, n_games: int, mcts_sims: int):
        games = []
        for _ in tqdm(range(n_games), desc="Self-play in progress..."):
            agents = [
                GiottoAgent(simulations=mcts_sims, valuenet=self.net),
                GiottoAgent(simulations=mcts_sims, valuenet=self.net),
            ]
            new_memory = self.simulate_game(agents)
            games.extend(new_memory)
        return games

    def train(self, epochs: int, games_per_epoch: int, mcts_sims: int):
        self.net.train()
        train_losses = []

        for epoch in range(epochs):
            replay_buffer = self.self_play(games_per_epoch, mcts_sims)
            random.shuffle(replay_buffer)
            total_loss = 0.0

            for state, result in replay_buffer:
                pred = self.net(state)

                target = torch.tensor([[result]], dtype=torch.float32)
                loss = F.mse_loss(pred, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(replay_buffer)
            train_losses.append(avg_loss)
            print(f"[Train] Epoch {epoch+1}/{epochs} - MSE: {avg_loss:.4f}")

        return train_losses

    @torch.no_grad()
    def test(self, n_games: int, mcts_sims: int, opp: GenericAgent):
        """Test by playing against random opponent."""
        self.net.eval()

        # vs random
        results = []
        agents = [GiottoAgent(simulations=mcts_sims, valuenet=self.net), opp]
        pbar = tqdm(
            range(n_games), desc=f"Playing vs {opp.name}", ncols=100, leave=True
        )
        for _ in range(n_games):
            env = self.base_env.clone()
            env.reset()

            while not env.done:
                action = agents[env.current_player].select_action(env)
                env.step(action)
            env_result = env.info["winner"]
            results.append(env_result)
            pbar.set_postfix(
                {
                    "W": results.count(0),
                    "D": results.count(-1),
                    "L": results.count(1),
                }
            )
        return results

    def save_metrics(
        self, path: str, train_losses: list, test_results: list, config: dict
    ):
        results = {
            "train_mse": train_losses,
            "test_results": test_results,
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
        train_games = n_games
        test_games = 100
        games_per_epoch = train_games // epochs

        print("Training ...")
        train_losses = self.train(epochs, games_per_epoch, mcts_sims)
        self.save_model(os.path.join(log_dir, "valuenet.pt"))

        print("Testing ...")
        test_results = self.test(test_games, mcts_sims, RandomAgent())
        print(
            f"[Test] vs Random: {test_results.count(0)} W, {test_results.count(-1)} D, {test_results.count(1)} L"
        )

        self.save_metrics(
            path=os.path.join(log_dir, "results_tris.json"),
            train_losses=train_losses,
            test_results=test_results,
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
    trainer.run(epochs=30, n_games=5000, mcts_sims=1000, log_dir=LOGS_DIR)
