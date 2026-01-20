import random
import os
import json
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from collections import deque
from giotto.agents.algorithms.value_net.value_net import ValueNet
from giotto.envs.generic import GenericEnv
from giotto.agents.generic import GenericAgent
from giotto.agents.random import RandomAgent
from giotto.agents.giotto import GiottoAgent
from giotto.utils.simmetries import EquivalentBoards


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
            new_memory.append((state, result))
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

    def augment_games(self, games: list) -> list:
        if hasattr(self.base_env, "simmetries") and isinstance(
            self.base_env.simmetries, EquivalentBoards
        ):
            augmented = []
            for state, result in games:
                board, player_id = state
                for eq_board in self.base_env.simmetries.get_equivalent_boards(board):
                    augmented.append(((eq_board, player_id), result))
            return augmented
        else:
            return games

    def train(
        self,
        epochs: int,
        games_per_epoch: int,
        steps_per_epoch: int,
        batch_size: int,
        mcts_sims: int,
        buffer_length: int,
    ):
        self.net.train()
        train_losses = []
        replay_buffer = deque(maxlen=buffer_length)

        for epoch in range(epochs):
            new_games = self.self_play(games_per_epoch, mcts_sims)
            augmented = self.augment_games(new_games)
            replay_buffer.extend(augmented)

            for step in range(steps_per_epoch):
                batch = random.sample(
                    replay_buffer, min(len(replay_buffer), batch_size)
                )

                self.optimizer.zero_grad()
                batch_loss = 0.0

                for state, result in batch:
                    value_input = self.net.process_state(state)
                    pred = self.net(value_input)
                    target = torch.tensor([[result]], dtype=torch.float32)

                    loss = F.mse_loss(pred, target)
                    batch_loss += loss

                batch_loss = batch_loss / len(batch)
                batch_loss.backward()
                self.optimizer.step()

                train_losses.append(batch_loss.item())
            print(f"[Train] Epoch {epoch+1}/{epochs} - MSE: {train_losses[-1]:.4f}")

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

    def run(
        self,
        epochs: int,
        steps_per_epoch: int,
        batch_size: int,
        n_games: int,
        mcts_sims: int,
        buffer_length: int,
        log_dir: str,
    ):
        train_games = n_games
        test_games = 100
        games_per_epoch = train_games // epochs

        print("Training ...")
        train_losses = self.train(
            epochs,
            games_per_epoch,
            steps_per_epoch,
            batch_size,
            mcts_sims,
            buffer_length,
        )
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
    trainer.run(
        epochs=30,
        steps_per_epoch=20,
        batch_size=64,
        n_games=1000,
        mcts_sims=500,
        buffer_length=300,
        log_dir=LOGS_DIR,
    )

# TODO:
# target is mcts value instead of game result
# batch norm and residual blocks ?
