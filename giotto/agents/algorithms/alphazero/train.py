from __future__ import annotations

import json
import logging
import os
import random
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from giotto.agents.algorithms.alphazero.mcts import AlphaZeroMCTS, AZNode
from giotto.agents.algorithms.alphazero.net import AlphaZeroNet
from giotto.envs.generic import GenericEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class AlphaZeroTrainer:
    """Full AlphaZero training pipeline.

    Generates self-play games and trains the network on results to emulate MCTS.
    Evaluates the network by scoring precision on labeled moves dataset generated separately with a solver.
    """

    def __init__(
        self,
        env: GenericEnv,
        config: dict,
        save_dir: str = "checkpoints",
        device: str = "cpu",
    ):
        self.base_env = env
        self.config = config
        net = AlphaZeroNet(
            input_size=self.config["network"]["input_size"],
            value_output_size=self.config["network"]["value_output_size"],
            policy_output_size=self.config["network"]["policy_output_size"],
            channels=self.config["network"]["channels"],
            residual_blocks=self.config["network"]["residual_blocks"],
        )
        self.net = net.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.config["training"]["learning_rate"])

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # ============================
        # Metrics
        # ============================
        self.metrics = {"iterations": []}

        # Save config immediately
        config_path = self.save_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(self.config, f)

    # ============================
    # Utility
    # ============================

    def save_model(self, name: str) -> None:
        """Save model weights."""
        path = self.save_dir / name
        torch.save(self.net.state_dict(), path)
        logging.info(f"Model saved to {path}")

    def load_model(self, name: str) -> None:
        """Load model weights."""
        path = self.save_dir / name
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def save_metrics(self) -> None:
        """Save metrics to disk."""
        metrics_path = self.save_dir / "metrics.yaml"
        with open(metrics_path, "w") as f:
            yaml.safe_dump(self.metrics, f)

    def save_plot_metrics(self):
        """Saves plots with metrics."""
        if not self.metrics["iterations"]:
            return

        iterations = [it["iteration"] for it in self.metrics["iterations"]]

        total_losses = [it.get("avg_total_loss") for it in self.metrics["iterations"]]
        policy_losses = [it.get("avg_policy_loss") for it in self.metrics["iterations"]]
        value_losses = [it.get("avg_value_loss") for it in self.metrics["iterations"]]

        def _extract_valid(metric_name: str):
            values = []
            for it in self.metrics["iterations"]:
                valid = it.get("validation_metrics")
                values.append(None if not valid else valid.get(metric_name))
            # matplotlib skips NaN nicely; None can be problematic -> convert to NaN
            return [np.nan if v is None else float(v) for v in values]

        val_policy_strong_acc = _extract_valid("policy_strong_acc")
        val_policy_weak_acc = _extract_valid("policy_weak_acc")
        val_value_sign_acc = _extract_valid("value_sign_accuracy")
        val_value_mse = _extract_valid("value_mse")
        val_value_mae = _extract_valid("value_mae")

        def _save(fig, file_name: str):
            fig.tight_layout()
            fig.savefig(self.save_dir / file_name, dpi=150)
            plt.close(fig)

        # Plot 1: losses
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(iterations, total_losses, marker="o", color="C0", label="Total loss")
        ax.plot(iterations, policy_losses, marker="o", color="C1", label="Policy loss")
        ax.plot(iterations, value_losses, marker="o", color="C2", label="Value loss")
        ax.set_title("Training losses")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save(fig, "train_losses.png")

        # Plot 2: accuracies
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(iterations, val_policy_strong_acc, marker="o", color="C0", label="Policy strong acc")
        ax.plot(iterations, val_policy_weak_acc, marker="o", color="C1", label="Policy weak acc")
        ax.plot(iterations, val_value_sign_acc, marker="o", color="C2", label="Value sign acc")
        ax.set_title("Validation accuracies")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save(fig, "val_accuracies.png")

        # Plot 3: MAE and MSE
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(iterations, val_value_mae, marker="o", color="C3", label="Value MAE")
        ax.plot(iterations, val_value_mse, marker="o", color="C4", label="Value MSE")
        ax.set_title("Validation value errors")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Error")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save(fig, "val_value_errors.png")

    # ============================
    # Self Play
    # ============================

    def build_policy_target_from_root(self, root: AZNode) -> np.ndarray:
        """Extracts the training targets from MCTS node."""
        visit_counts = np.zeros(self.net.policy_output_size, dtype=np.float32)

        for action_index_1based, child_node in root.children.items():
            action_index_0based = action_index_1based - 1
            visit_counts[action_index_0based] = float(child_node.n_visits)

        total_visits = float(visit_counts.sum())
        policy_target = visit_counts / total_visits
        return policy_target

    def self_play_game(self) -> list:
        """Generate one self-play game."""
        self.net.eval()
        episode_records = []

        env = self.base_env.clone()
        env.reset()
        move_count = 0
        while not env.done:
            schedule = self.config["mcts"]["temperature_schedule"]
            temperature = self.config["mcts"]["temperature"] if move_count < schedule else 0.0

            mcts = AlphaZeroMCTS(
                self.net,
                n_simulations=self.config["mcts"]["n_sims"],
                cpuct=self.config["mcts"]["cpuct"],
                dirichlet_alpha=self.config["mcts"]["dirichlet_alpha"],
                dirichlet_eps=self.config["mcts"]["dirichlet_epsilon"],
                temperature=temperature,
            )
            action, root = mcts.run(env)
            policy_target = self.build_policy_target_from_root(root)
            episode_records.append(
                {
                    "state": env.get_state(),
                    "policy_target": policy_target,
                }
            )
            env.step(action)
            move_count += 1

        winner = env.info["winner"]
        for record in episode_records:
            if winner == -1:
                value_target = 0.0
            else:
                value_target = 1.0 if winner == record["state"][1] else -1.0
            record["value_target"] = value_target

        self.net.train()
        return episode_records

    def self_play(self, n_games: int) -> list[dict]:
        """Generate dataset via self-play."""
        data = []
        for _ in tqdm(range(n_games), desc="Self-play"):
            data.extend(self.self_play_game())
        return data

    def augment_data(self, data: list[dict]) -> list:
        """Augment moves-target dataset using env.simmetries properties."""
        if not hasattr(self.base_env, "simmetries"):
            logger.warning("No env.simmetries found. Proceding without augmentations.")
            return data

        augmented = []
        for move in data:
            move_augmentations = self.base_env.simmetries.get_equivalent_boards(move["state"][0], move["policy_target"])
            for aug in move_augmentations:
                augmented.append(
                    {
                        "state": [aug[0], move["state"][1]],  # reconstruct state
                        "policy_target": aug[1],
                        "value_target": move["value_target"],
                    }
                )

        return augmented

    # ============================
    # Validation
    # ============================
    def validate(self, jsonl_path: str | Path, max_lines: int | None = None, draw_band: float = 0.1) -> dict:
        """Validate the net against a JSONL dataset with fields:
        - state: [board, player_id]
        - strong_best_moves: [int, ...]   (1-based)
        - gto_result: -1/0/1.
        """
        self.net.eval()

        total_positions = 0
        strong_correct = 0
        weak_correct = 0
        value_mse = 0.0
        value_mae = 0.0
        value_sign_correct = 0

        try:
            with open(jsonl_path, encoding="utf-8") as file:
                for _idx, line in enumerate(file, start=1):
                    if max_lines is not None and total_positions >= max_lines:
                        break

                    line = line.strip()  # noqa: PLW2901
                    if not line:
                        continue

                    record = json.loads(line)

                    board_list, player_id = record["state"]
                    board_array = np.asarray(board_list, dtype=np.int64)
                    state = [board_array, int(player_id)]

                    strong_best_moves = record.get("strong_best_moves", None)
                    weak_best_moves = record.get("weak_best_moves", None)
                    gto_result = record.get("gto_result", None)
                    if strong_best_moves is None or weak_best_moves is None or gto_result is None:
                        continue

                    policy, value = self.net.predict(state)

                    total_positions += 1
                    # policy metrics
                    best_action = policy.argmax() + 1
                    if best_action in strong_best_moves:
                        strong_correct += 1
                    if best_action in weak_best_moves:
                        weak_correct += 1
                    # value metrics
                    value_error = value - gto_result
                    value_mse += value_error * value_error
                    value_mae += np.abs(value_error)
                    if (np.abs(value) < draw_band and gto_result == 0) or np.sign(value) == np.sign(gto_result):
                        value_sign_correct += 1
        except Exception as e:
            logging.warning(f"Running validation failed: {e}")
            return None

        if total_positions == 0:
            return {"total_positions": 0}

        return {
            "total_positions": int(total_positions),
            "policy_strong_acc": float(strong_correct / total_positions),
            "policy_weak_acc": float(weak_correct / total_positions),
            "value_mse": float(value_mse / total_positions),
            "value_mae": float(value_mae / total_positions),
            "value_sign_accuracy": float(value_sign_correct / total_positions),
        }

    # ============================
    # Training
    # ============================

    def train_step(self, batch_records: list[dict]) -> dict:
        """One optimizer step on a batch of replay records."""
        self.net.train()

        state_tensors = []
        policy_targets = []
        value_targets = []

        for record in batch_records:
            state = record["state"]
            policy_target = record["policy_target"]
            value_target = record["value_target"]

            state_tensors.append(self.net.process_state(state))  # (1, C, H, W)
            policy_targets.append(np.asarray(policy_target, dtype=np.float32))
            value_targets.append(float(value_target))

        states = torch.cat(state_tensors, dim=0).to(self.device)  # (B, C, H, W)
        policy_targets_tensor = torch.from_numpy(np.stack(policy_targets, axis=0)).to(self.device)  # (B, A)
        value_targets_tensor = torch.tensor(value_targets, dtype=torch.float32, device=self.device).unsqueeze(
            1
        )  # (B, 1)

        policy_logits, value_predictions = self.net(states)  # logits: (B,A), values: (B,1)

        log_policy = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policy_targets_tensor * log_policy).sum(dim=1).mean()  # cross-entropy with soft targets

        value_loss = F.mse_loss(value_predictions, value_targets_tensor)  # (z - v)^2

        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }

    def train(self):
        """Full AlphaZero training loop."""
        iterations = int(self.config["training"]["iterations"])
        buffer_size = int(self.config["training"]["buffer_size"])
        games_per_iteration = int(self.config["training"]["games_per_iteration"])
        batch_size = int(self.config["training"]["batch_size"])

        # TODO: check this value
        training_steps_per_iteration = int(self.config["training"].get("training_steps_per_iteration", 0))

        replay_buffer: deque[dict] = deque(maxlen=buffer_size)

        for iteration_index in range(iterations):
            logger.info("========== Iteration %d/%d ==========", iteration_index + 1, iterations)

            # -------------------
            # Self-play collection
            # -------------------
            new_records: list[dict] = []
            for _ in tqdm(range(games_per_iteration), desc="Self-play"):
                new_records.extend(self.self_play_game())

            new_records = self.augment_data(new_records)
            replay_buffer.extend(new_records)

            if len(replay_buffer) < batch_size:
                logger.info("Not enough samples to train yet (need %d).", batch_size)
                continue

            # -------------------
            # Network training
            # -------------------
            if training_steps_per_iteration <= 0:
                training_steps_per_iteration = max(1, len(replay_buffer) // batch_size)

            losses_total = []
            losses_policy = []
            losses_value = []

            for _ in tqdm(range(training_steps_per_iteration), desc="Training"):
                batch_records = random.sample(list(replay_buffer), batch_size)
                loss_dict = self.train_step(batch_records)
                losses_total.append(loss_dict["total_loss"])
                losses_policy.append(loss_dict["policy_loss"])
                losses_value.append(loss_dict["value_loss"])

            avg_total = float(np.mean(losses_total))
            avg_policy = float(np.mean(losses_policy))
            avg_value = float(np.mean(losses_value))

            logger.info("Avg loss: total=%.4f policy=%.4f value=%.4f", avg_total, avg_policy, avg_value)

            # -------------------
            # Checkpoint
            # -------------------
            self.save_model(f"checkpoint_iter_{iteration_index + 1}.pt")

            # -------------------
            # Gating
            # -------------------
            # TODO: implement model gating
            # accepted = False
            # if win_rate > 0.55:
            #     accepted = True
            #     best_net = copy.deepcopy(self.net)
            #     self.save_model("best_model.pt")
            #     logger.info("New model accepted.")
            # else:
            #     self.net.load_state_dict(best_net.state_dict())
            #     logger.info("New model rejected; reverted to best.")

            # -------------------
            # Validation
            # -------------------
            valid_metrics = None
            if self.config["eval"]["run_eval"]:
                valid_metrics = self.validate(
                    jsonl_path=Path(__file__).parent / self.config["eval"]["eval_dataset"], draw_band=0.1
                )

            # -------------------
            # Metrics
            # -------------------
            self.metrics["iterations"].append(
                {
                    "iteration": iteration_index + 1,
                    "replay_buffer_size": len(replay_buffer),
                    "avg_total_loss": float(avg_total),
                    "avg_policy_loss": float(avg_policy),
                    "avg_value_loss": float(avg_value),
                    # "gating_win_rate": win_rate,
                    # "accepted": accepted,
                    "validation_metrics": valid_metrics,
                }
            )
            self.save_metrics()
            self.save_plot_metrics()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AlphaZero Net training")
    parser.add_argument("-g", "--game", help="game to play [tris, connect4]", required=True)
    parser.add_argument("-c", "--config", help="path to config yaml file", required=False)
    args = vars(parser.parse_args())

    if args["game"].lower() == "tris":
        from giotto.envs.tris import TrisEnv

        env = TrisEnv()
        with open("config_tris.yaml") as f:
            config = yaml.safe_load(f)
    elif args["game"].lower() == "connect4":
        from giotto.envs.connect4 import Connect4Env

        env = Connect4Env()
        with open("config_c4.yaml") as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"{args['game']} not a valid game")

    if args["config"]:
        try:
            with open(args["config"]) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise FileExistsError(f"Failed to load config file. {e}")  # noqa: B904

    LOGS_DIR = (
        Path(__file__).parent
        / ".."
        / ".."
        / ".."
        / ".."
        / "logs"
        / "alphazero"
        / args["game"]
        / datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    )
    os.makedirs(LOGS_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = AlphaZeroTrainer(
        env=env,
        config=config,
        save_dir=LOGS_DIR,
        device=device,
    )

    trainer.train()
