from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import random
import traceback
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
import torch.multiprocessing as mp

from giotto.agents.algorithms.alphazero.mcts import AlphaZeroMCTS, AZNode
from giotto.agents.algorithms.alphazero.net import AlphaZeroNet
from giotto.agents.alphazero import AlphaZeroAgent
from giotto.agents.mcts import MCTSAgent
from giotto.envs.generic import GenericEnv
from giotto.utils.text_play import play_n_games

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _set_worker_threads(num_processes: int) -> None:
    """Avoid CPU oversubscription when spawning multiple processes.

    PyTorch warns that each subprocess may try to use all vCPUs (via intra-op threads),
    so cap torch threads per process to roughly floor(N/M).
    """
    try:
        n_cpus = os.cpu_count() or 1
        threads = max(1, int(math.floor(n_cpus / max(1, num_processes))))
        torch.set_num_threads(threads)
    except Exception:
        pass


def _run_self_play_game(
    net: AlphaZeroNet,
    base_env: GenericEnv,
    config: dict,
) -> list[dict]:
    """Generate one self-play game with MCTS tree reuse between moves.

    Returns a list of {state, policy_target, value_target} records.
    """
    env = base_env.clone()
    env.reset()
    move_count = 0
    episode_records = []

    mcts_batch_size = config["mcts"].get("batch_size", 1)
    mcts = AlphaZeroMCTS(
        net,
        n_simulations=config["mcts"]["n_sims"],
        cpuct=config["mcts"]["cpuct"],
        dirichlet_alpha=config["mcts"]["dirichlet_alpha"],
        dirichlet_eps=config["mcts"]["dirichlet_epsilon"],
        batch_size=mcts_batch_size,
    )

    while not env.done:
        schedule = config["mcts"]["temperature_schedule"]
        temperature = config["mcts"]["temperature"] if move_count < schedule else 0.0

        if mcts_batch_size > 1:
            action, root = mcts.run_batched(env, temperature=temperature)
        else:
            action, root = mcts.run(env, temperature=temperature)

        # Build policy target from root visit counts
        visit_counts = np.zeros(net.policy_output_size, dtype=np.float32)
        for action_idx_1based, child in root.children.items():
            visit_counts[action_idx_1based - 1] = float(child.n_visits)
        policy_target = visit_counts / visit_counts.sum()

        episode_records.append({"state": env.get_state(), "policy_target": policy_target})
        env.step(action)
        mcts.advance_root(action)  # reuse subtree for next move
        move_count += 1

    winner = env.info["winner"]
    for record in episode_records:
        if winner == -1:
            record["value_target"] = 0.0
        else:
            record["value_target"] = 1.0 if winner == record["state"][1] else -1.0

    return episode_records


def _self_play_worker(
    worker_id: int,
    n_games: int,
    base_env: GenericEnv,
    config: dict,
    device: str,
    net_state_dict: dict,
    out_queue: mp.SimpleQueue,
    num_processes: int,
    seed: int | None = None,
) -> None:
    """Worker that generates self-play data and pushes it to out_queue."""
    try:
        _set_worker_threads(num_processes)

        if seed is not None:
            random.seed(seed + worker_id)
            np.random.seed(seed + worker_id)
            torch.manual_seed(seed + worker_id)

        net = AlphaZeroNet(
            input_size=config["network"]["input_size"],
            value_output_size=config["network"]["value_output_size"],
            policy_output_size=config["network"]["policy_output_size"],
            channels=config["network"]["channels"],
            residual_blocks=config["network"]["residual_blocks"],
        )
        net.load_state_dict(net_state_dict, strict=True)
        net = net.to(device)
        net.eval()

        local_data = []
        for _ in range(int(n_games)):
            local_data.extend(_run_self_play_game(net, base_env, config))

        out_queue.put(("ok", worker_id, local_data))

    except Exception as e:
        tb = traceback.format_exc()
        out_queue.put(("err", worker_id, f"{e}\n{tb}"))


def _eval_worker(
    worker_id: int,
    n_games: int,
    base_env: GenericEnv,
    config: dict,
    device: str,
    net_state_dict: dict,
    out_queue: mp.SimpleQueue,
    num_processes: int,
    seed: int | None = None,
) -> None:
    """Worker that plays evaluation matches vs MCTS and returns aggregated rates."""
    try:
        _set_worker_threads(num_processes)

        if seed is not None:
            random.seed(seed + 10_000 + worker_id)
            np.random.seed(seed + 10_000 + worker_id)
            torch.manual_seed(seed + 10_000 + worker_id)

        net = AlphaZeroNet(
            input_size=config["network"]["input_size"],
            value_output_size=config["network"]["value_output_size"],
            policy_output_size=config["network"]["policy_output_size"],
            channels=config["network"]["channels"],
            residual_blocks=config["network"]["residual_blocks"],
        )
        net.load_state_dict(net_state_dict, strict=True)
        net = net.to(device)
        net.eval()

        az_agent = AlphaZeroAgent(
            game=config["game"],
            simulations=config["mcts"]["n_sims"],
            cpuct=config["mcts"]["cpuct"],
            net=net,
        )
        mcts_agent = MCTSAgent(
            simulations=config["mcts"]["n_sims"],
            cpuct=config["mcts"]["cpuct"],
        )
        agents = [az_agent, mcts_agent]

        winners = play_n_games(int(n_games), base_env.clone(), agents, invert_starts=True)
        out_queue.put(("ok", worker_id, winners))

    except Exception as e:
        tb = traceback.format_exc()
        out_queue.put(("err", worker_id, f"{e}\n{tb}"))


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

    def load_metrics(self) -> None:
        """Load metrics from disk, merging with the current empty structure."""
        metrics_path = self.save_dir / "metrics.yaml"
        if metrics_path.exists():
            with open(metrics_path) as f:
                loaded = yaml.safe_load(f)
            if loaded:
                self.metrics = loaded

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

        # Plot 4: match vs MCTS
        def _extract_test(metric_name: str):
            values = []
            for it in self.metrics["iterations"]:
                t = it.get("match_metrics")
                values.append(None if not t else t.get(metric_name))
            return [np.nan if v is None else float(v) for v in values]

        az_name = "AlphaZeroAgent"
        mcts_name = "MCTSAgent"

        test_az_win = _extract_test(az_name)
        test_mcts_win = _extract_test(mcts_name)
        test_draw = _extract_test("Draw")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(iterations, test_az_win, marker="o", label=f"{az_name} win rate")
        ax.plot(iterations, test_mcts_win, marker="o", label=f"{mcts_name} win rate")
        ax.plot(iterations, test_draw, marker="o", label="Draw rate")
        ax.set_title("Test vs MCTS (no-net baseline)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save(fig, "test_vs_mcts_rates.png")

    # ============================
    # Self Play
    # ============================

    def build_policy_target_from_root(self, root: AZNode) -> np.ndarray:
        """Extracts the training targets from MCTS node."""
        visit_counts = np.zeros(self.net.policy_output_size, dtype=np.float32)
        for action_index_1based, child_node in root.children.items():
            visit_counts[action_index_1based - 1] = float(child_node.n_visits)
        return visit_counts / visit_counts.sum()

    def self_play_game(self) -> list:
        """Generate one self-play game with MCTS tree reuse between moves."""
        self.net.eval()
        records = _run_self_play_game(self.net, self.base_env, self.config)
        self.net.train()
        return records

    def self_play(self, n_games: int) -> list[dict]:
        """Generate dataset via self-play (single process)."""
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
                        "state": [aug[0], move["state"][1]],
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
        net = self.net
        net.eval()

        states: list = []
        strong_moves_list: list = []
        weak_moves_list: list = []
        gto_results: list = []

        try:
            with open(jsonl_path, encoding="utf-8") as file:
                for line in file:
                    if max_lines is not None and len(states) >= max_lines:
                        break
                    line = line.strip()  # noqa: PLW2901
                    if not line:
                        continue
                    record = json.loads(line)
                    strong_best_moves = record.get("strong_best_moves")
                    weak_best_moves = record.get("weak_best_moves")
                    gto_result = record.get("gto_result")
                    if strong_best_moves is None or weak_best_moves is None or gto_result is None:
                        continue
                    board_list, player_id = record["state"]
                    states.append([np.asarray(board_list, dtype=np.int64), int(player_id)])
                    strong_moves_list.append(strong_best_moves)
                    weak_moves_list.append(weak_best_moves)
                    gto_results.append(gto_result)
        except Exception as e:
            logging.warning(f"Running validation failed: {e}")
            return None

        total_positions = len(states)
        if total_positions == 0:
            return {"total_positions": 0}

        # Batch predict in chunks of 256
        batch_size = 256
        all_policies: list[np.ndarray] = []
        all_values: list[np.ndarray] = []
        for i in range(0, total_positions, batch_size):
            chunk = states[i : i + batch_size]
            policies, values = net.batch_predict(chunk)
            all_policies.append(policies)
            all_values.append(values)

        policies_arr = np.concatenate(all_policies, axis=0)  # (N, A)
        values_arr = np.concatenate(all_values, axis=0)  # (N,)
        gto_arr = np.array(gto_results, dtype=np.float32)

        best_actions = policies_arr.argmax(axis=1) + 1  # 1-based
        strong_correct = sum(int(a in s) for a, s in zip(best_actions, strong_moves_list))
        weak_correct = sum(int(a in w) for a, w in zip(best_actions, weak_moves_list))

        value_errors = values_arr - gto_arr
        value_mse = float(np.mean(value_errors**2))
        value_mae = float(np.mean(np.abs(value_errors)))
        value_sign_correct = int(
            np.sum((np.abs(values_arr) < draw_band) & (gto_arr == 0) | (np.sign(values_arr) == np.sign(gto_arr)))
        )

        return {
            "total_positions": int(total_positions),
            "policy_strong_acc": float(strong_correct / total_positions),
            "policy_weak_acc": float(weak_correct / total_positions),
            "value_mse": value_mse,
            "value_mae": value_mae,
            "value_sign_accuracy": float(value_sign_correct / total_positions),
        }

    def test_vs_mcts(self, n_games: int):
        """Plays a number of games against MCTS without network to evaluate result."""
        net = self.net
        net.eval()

        az_agent = AlphaZeroAgent(
            game=self.config["game"],
            simulations=self.config["mcts"]["n_sims"],
            cpuct=self.config["mcts"]["cpuct"],
            net=net,
        )
        mcts_agent = MCTSAgent(
            simulations=self.config["mcts"]["n_sims"],
            cpuct=self.config["mcts"]["cpuct"],
        )
        agents = [az_agent, mcts_agent]

        winners = play_n_games(n_games, self.base_env.clone(), agents, invert_starts=True)

        test_results = {}
        for agent in agents:
            test_results[agent.name] = winners[agent.name] / float(n_games)
        test_results["Draw"] = winners["Draw"] / float(n_games)
        return test_results

    # ============================
    # Training
    # ============================

    def train_step(self, batch_records: list[dict]) -> dict:
        """One optimizer step on a batch of replay records (vectorized state processing)."""
        self.net.train()

        # Vectorized state → tensor conversion
        boards = np.stack([r["state"][0] for r in batch_records])  # (B, H, W)
        players = np.array([r["state"][1] for r in batch_records])  # (B,)
        current = (boards == players[:, None, None]).astype(np.float32)  # (B, H, W)
        opponent = (boards == (1 - players)[:, None, None]).astype(np.float32)
        x = np.stack([current, opponent], axis=1)  # (B, 2, H, W)

        states = torch.from_numpy(x).to(self.device)
        policy_targets_tensor = torch.from_numpy(np.stack([r["policy_target"] for r in batch_records])).to(self.device)
        value_targets_tensor = torch.tensor(
            [r["value_target"] for r in batch_records], dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        policy_logits, value_predictions = self.net(states)

        log_policy = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policy_targets_tensor * log_policy).sum(dim=1).mean()
        value_loss = F.mse_loss(value_predictions, value_targets_tensor)
        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }

    def self_play_parallel(self, n_games: int, num_workers: int, seed: int | None = None) -> list[dict]:
        """Generate dataset via self-play using multiprocessing (CPU)."""
        n_games = int(n_games)
        num_workers = max(1, int(num_workers))
        if n_games <= 0:
            return []

        # Snapshot weights once; workers load their own copy.
        net = self.net
        net_state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}

        base = n_games // num_workers
        rem = n_games % num_workers
        worker_games = [base + (1 if i < rem else 0) for i in range(num_workers)]
        worker_games = [g for g in worker_games if g > 0]
        num_workers = len(worker_games)

        out_q: mp.SimpleQueue = mp.SimpleQueue()
        procs: list[mp.Process] = []

        for wid, ng in enumerate(worker_games):
            p = mp.Process(
                target=_self_play_worker,
                args=(wid, ng, self.base_env, self.config, "cpu", net_state_dict, out_q, num_workers, seed),
            )
            p.daemon = False
            p.start()
            procs.append(p)

        collected = []
        for _ in tqdm(range(num_workers), desc="Self-play (mp)"):
            status, wid, payload = out_q.get()
            if status != "ok":
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                for p in procs:
                    p.join(timeout=1.0)
                raise RuntimeError(f"Self-play worker {wid} failed:\n{payload}")
            collected.extend(payload)

        for p in procs:
            p.join()

        return collected

    def test_vs_mcts_parallel(self, n_games: int, num_workers: int = 4, seed: int | None = 5678):
        """Plays evaluation matches vs MCTS using multiprocessing."""
        n_games = int(n_games)
        num_workers = max(1, int(num_workers))
        if n_games <= 0:
            return {}

        net = self.net
        net_state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}

        base = n_games // num_workers
        rem = n_games % num_workers
        worker_games = [base + (1 if i < rem else 0) for i in range(num_workers)]
        worker_games = [g for g in worker_games if g > 0]
        num_workers = len(worker_games)

        out_q: mp.SimpleQueue = mp.SimpleQueue()
        procs: list[mp.Process] = []

        for wid, ng in enumerate(worker_games):
            p = mp.Process(
                target=_eval_worker,
                args=(wid, ng, self.base_env, self.config, "cpu", net_state_dict, out_q, num_workers, seed),
            )
            p.daemon = False
            p.start()
            procs.append(p)

        totals = {}
        total_games_done = 0

        for _ in tqdm(range(num_workers), desc="Eval vs MCTS (mp)"):
            status, wid, payload = out_q.get()
            if status != "ok":
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                for p in procs:
                    p.join(timeout=1.0)
                raise RuntimeError(f"Eval worker {wid} failed:\n{payload}")

            winners = payload
            for k, v in winners.items():
                totals[k] = totals.get(k, 0) + int(v)
            total_games_done += sum(int(v) for v in winners.values())

        for p in procs:
            p.join()

        return {k: v / float(n_games) for k, v in totals.items()}

    def train(self, start_iteration: int = 0):
        """Full AlphaZero training loop.

        Args:
            start_iteration: Iteration index to resume from (0-based). Iterations before
                this value are skipped; their metrics are expected to already be in self.metrics.
        """
        iterations = int(self.config["training"]["iterations"])
        buffer_size = int(self.config["training"]["buffer_size"])
        games_per_iteration = int(self.config["training"]["games_per_iteration"])
        batch_size = int(self.config["training"]["batch_size"])
        training_steps_per_iteration = int(self.config["training"].get("training_steps_per_iteration", 0))

        replay_buffer: deque[dict] = deque(maxlen=buffer_size)

        num_self_play_workers = int(self.config["training"].get("n_play_workers", max(1, (os.cpu_count() or 2) // 2)))
        num_eval_workers = int(self.config["eval"].get("n_workers", max(1, (os.cpu_count() or 2) // 2)))

        if start_iteration > 0:
            logger.info("Resuming from iteration %d (replay buffer starts empty).", start_iteration + 1)

        for iteration_index in range(start_iteration, iterations):
            logger.info("========== Iteration %d/%d ==========", iteration_index + 1, iterations)

            # -------------------
            # Self-play collection
            # -------------------
            if num_self_play_workers > 1 and games_per_iteration > 1:
                new_records = self.self_play_parallel(games_per_iteration, num_workers=num_self_play_workers)
            else:
                new_records = []
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
            steps = (
                training_steps_per_iteration
                if training_steps_per_iteration > 0
                else max(1, len(replay_buffer) // batch_size)
            )

            # Snapshot buffer once to avoid repeated O(N) deque→list conversion
            buffer_list = list(replay_buffer)

            losses_total = []
            losses_policy = []
            losses_value = []

            for _ in tqdm(range(steps), desc="Training"):
                batch_records = random.sample(buffer_list, batch_size)
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
            # Validation
            # -------------------
            valid_metrics = None
            if self.config["eval"]["run_eval"]:
                valid_metrics = self.validate(
                    jsonl_path=Path(__file__).parent / self.config["eval"]["eval_dataset"], draw_band=0.1
                )

            if self.config["eval"]["play_vs_mcts"]:
                if num_eval_workers > 1 and int(self.config["eval"]["n_matches"]) > 1:
                    match_metrics = self.test_vs_mcts_parallel(
                        self.config["eval"]["n_matches"], num_workers=num_eval_workers
                    )
                else:
                    match_metrics = self.test_vs_mcts(self.config["eval"]["n_matches"])
            else:
                match_metrics = {}

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
                    "match_metrics": match_metrics,
                    "validation_metrics": valid_metrics,
                }
            )
            self.save_metrics()
            self.save_plot_metrics()


def _make_env(game: str) -> GenericEnv:
    """Instantiate a game environment by name."""
    if game.lower() == "tris":
        from giotto.envs.tris import TrisEnv

        return TrisEnv()
    elif game.lower() == "connect4":
        from giotto.envs.connect4 import Connect4Env

        return Connect4Env()
    else:
        raise ValueError(f"{game!r} is not a valid game. Choose 'tris' or 'connect4'.")


if __name__ == "__main__":
    import argparse

    with contextlib.suppress(RuntimeError):
        mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="AlphaZero Net training")
    parser.add_argument("-g", "--game", help="game to play [tris, connect4]", required=False)
    parser.add_argument("-c", "--config", help="path to config yaml file", required=False)
    parser.add_argument(
        "--resume",
        help="path to a previous log directory to resume training from",
        required=False,
    )
    args = vars(parser.parse_args())

    start_iteration = 0

    if args["resume"]:
        resume_dir = Path(args["resume"]).resolve()
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

        config_path = resume_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.yaml found in resume directory: {resume_dir}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        game_name = config.get("game", args.get("game"))
        if not game_name:
            raise ValueError("Could not determine game from config. Provide --game explicitly.")
        env = _make_env(game_name)

        # Find the latest checkpoint in the resume directory
        checkpoints = sorted(
            resume_dir.glob("checkpoint_iter_*.pt"),
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {resume_dir}. Nothing to resume.")

        latest_checkpoint = checkpoints[-1]
        start_iteration = int(latest_checkpoint.stem.rsplit("_", 1)[-1])
        logger.info("Resuming from %s (iteration %d)", latest_checkpoint.name, start_iteration)

        LOGS_DIR = resume_dir

    else:
        if not args["game"]:
            parser.error("--game is required when not using --resume")

        game_name = args["game"]
        env = _make_env(game_name)

        default_config_name = "config_tris.yaml" if game_name.lower() == "tris" else "config_c4.yaml"
        with open(Path(__file__).parent / default_config_name) as f:
            config = yaml.safe_load(f)

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
            / game_name
            / datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
        os.makedirs(LOGS_DIR)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    trainer = AlphaZeroTrainer(
        env=env,
        config=config,
        save_dir=LOGS_DIR,
        device=device,
    )

    if args["resume"]:
        trainer.load_model(latest_checkpoint.name)
        trainer.load_metrics()

    trainer.train(start_iteration=start_iteration)
