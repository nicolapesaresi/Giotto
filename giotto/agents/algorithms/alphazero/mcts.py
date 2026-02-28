from __future__ import annotations

import math

import numpy as np

from giotto.envs.generic import GenericEnv

# try except block to make it work in browser mode
try:
    BROWSER_MODE = False
    from giotto.agents.algorithms.alphazero.net import AlphaZeroNet
except ImportError:
    BROWSER_MODE = True
    from giotto.agents.algorithms.alphazero.net_numpy import AlphaZeroNetNumpy as AlphaZeroNet


class AZNode:
    """Node in AlphaZero MCTS tree."""

    def __init__(
        self,
        env: GenericEnv,
        parent: AZNode | None = None,
        parent_action: int | None = None,
    ):
        self.parent = parent
        self.parent_action = parent_action
        self.env = env
        self.to_play = env.current_player

        self.prob = 0.0
        self.children = {}
        self.n_visits = 0
        self.total_score = 0

    def terminal_node_eval(self, env: GenericEnv, player: int):
        """Evaluates terminal nodes for backpropagation."""
        winner = env.info["winner"]
        if winner == -1:
            return 0
        elif winner == player:
            return 1
        else:
            return -1

    def expand(self):
        """Create child nodes for all valid actions. If state is terminal, evaluate and set the node's value."""
        if self.env.done:
            self.total_score = self.terminal_node_eval(self.env, self.to_play)
            return

        valid_actions = self.env.get_valid_actions()
        for action in valid_actions:
            child_env = self.env.clone()
            child_env.step(action)
            self.children[action] = AZNode(env=child_env, parent=self, parent_action=action)

    def select_child(self, cpuct: float):
        """Select the child node with the highest PUCT score."""
        best_puct = -np.inf
        best_child = None
        for child in self.children.values():
            puct = self.calculate_puct(child, cpuct)
            if puct > best_puct:
                best_puct = puct
                best_child = child
        return best_child

    def calculate_puct(self, child: AZNode, cpuct: float):
        """Calculate the PUCT score for a given child node."""
        # flip sign because perspective of child is always opposite of father
        exploitation_term = -child.avg_value
        exploration_term = child.prob * math.sqrt(self.n_visits) / (child.n_visits + 1)
        return exploitation_term + cpuct * exploration_term

    def backpropagate(self, value: int):
        """Update the current node and its ancestors with the given value."""
        self.total_score += value
        self.n_visits += 1
        if self.parent is not None:
            # alternate sign as player alternates
            self.parent.backpropagate(-value)

    def is_leaf(self):
        """Check if the node is a leaf (no children)."""
        return len(self.children) == 0

    def is_terminal(self):
        """Check if the node represents a terminal state."""
        return self.env.done

    @property
    def avg_value(self):
        """Calculate the average value of this node."""
        if self.n_visits == 0:
            return 0
        return self.total_score / self.n_visits

    def __str__(self):
        """Return a string containing the node's relevant information for debugging purposes."""
        return (
            f"Prob: {self.prob}\nTo play: {self.to_play}"
            + f"\nNumber of children: {len(self.children)}\nNumber of visits: {self.n_visits}"
            + f"\nAvg value: {self.avg_value}"
        )


class AlphaZeroMCTS:
    """AlphaZero style Monte Carlo Tree Search."""

    def __init__(
        self,
        net: AlphaZeroNet,
        n_simulations: int,
        cpuct: float,
        dirichlet_alpha: float = 1.0,
        dirichlet_eps: float | None = None,
        temperature: float | None = None,
    ):
        """Intantiates Monte Carlo Tree Search with a given neural network and configuration."""
        self.net = net
        self.n_simulations = n_simulations
        self.cpuct = cpuct

        self.skip_dirichlet = dirichlet_alpha is None or dirichlet_eps is None
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.temperature = temperature

    def run(self, env: GenericEnv):
        """Runs MCTS to select action."""
        self.net.eval()

        root_env = env.clone()
        root = AZNode(env=root_env, parent=None, parent_action=None)
        # ROOT EXPANSION
        # expand root, adding dirichlet noise to each action
        valid_actions = root_env.get_valid_actions()
        policy, value = self.net.predict(root_env.get_state())
        if not self.skip_dirichlet:
            noise = np.random.dirichlet([self.dirichlet_alpha] * self.net.policy_output_size)
            action_probs = ((1 - self.dirichlet_eps) * policy) + self.dirichlet_eps * noise
        else:
            action_probs = policy
        # valid action masking
        action_probs = [action_probs[a - 1] for a in valid_actions]
        action_probs /= np.sum(action_probs)  # reapply softmax

        for action, prob in zip(valid_actions, action_probs, strict=True):
            new_env = root_env.clone()
            new_env.step(action)
            root.children[action] = AZNode(env=new_env, parent=root, parent_action=action)
            root.children[action].prob = prob
        # root.n_visits = 1
        # root.total_score = value

        # SEARCH
        for _ in range(self.n_simulations):
            current_node = root

            # SELECTION
            while not current_node.is_terminal() and not current_node.is_leaf():
                current_node = current_node.select_child(cpuct=self.cpuct)

            # EXPANSION
            if not current_node.is_terminal():
                current_node.expand()
                policy, value = self.net.predict(current_node.env.get_state())
                valid_actions = current_node.env.get_valid_actions()
                action_probs = [policy[a - 1] for a in valid_actions]
                action_probs /= np.sum(action_probs)  # reapply softmax

                for action, prob in zip(valid_actions, action_probs, strict=True):
                    current_node.children[action].prob = prob
            # if node is terminal, get the value of it from env
            else:
                value = current_node.terminal_node_eval(env=current_node.env, player=current_node.to_play)

            # BACKPROPAGATION
            current_node.backpropagate(value)

        # select action (temperature regulates exploration)
        return self.select_action(root, self.temperature), root

    def select_action(self, root: AZNode, temperature: float = 0.0):
        """Select an action from the root based on visit counts, adjusted by temperature. Set 0 for greedy selection."""
        action_counts = {key: val.n_visits for key, val in root.children.items()}
        if temperature is None or temperature == 0.0:
            return max(action_counts, key=action_counts.get)
        elif temperature == np.inf:
            return np.random.choice(list(action_counts.keys()))
        else:
            distribution = np.array([*action_counts.values()]) ** (1 / temperature)
            return np.random.choice([*action_counts.keys()], p=distribution / sum(distribution))
