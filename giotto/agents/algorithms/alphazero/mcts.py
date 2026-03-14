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
    """Node in AlphaZero MCTS tree.

    Environments are materialized lazily: child nodes only clone the parent env
    when first accessed, reducing redundant clones for unexplored branches.
    """

    __slots__ = (
        "parent",
        "parent_action",
        "_env",
        "_source_env",
        "_source_action",
        "to_play",
        "prob",
        "children",
        "n_visits",
        "total_score",
    )

    def __init__(
        self,
        env: GenericEnv | None = None,
        parent: AZNode | None = None,
        parent_action: int | None = None,
        *,
        _source_env: GenericEnv | None = None,
        _source_action: int | None = None,
    ):
        self.parent = parent
        self.parent_action = parent_action
        self._env = env
        self._source_env = _source_env
        self._source_action = _source_action
        self.to_play = env.current_player if env is not None else None
        self.prob = 0.0
        self.children: dict[int, AZNode] = {}
        self.n_visits = 0
        self.total_score = 0.0

    @property
    def env(self) -> GenericEnv:
        """Return the environment, cloning from the parent lazily on first access."""
        if self._env is None:
            env = self._source_env.clone()
            env.step(self._source_action)
            self._env = env
            self.to_play = env.current_player
            self._source_env = None
            self._source_action = None
        return self._env

    @property
    def avg_value(self) -> float:
        """Average value of the node."""
        if self.n_visits == 0:
            return 0.0
        return self.total_score / self.n_visits

    def terminal_node_eval(self, env: GenericEnv, player: int) -> float:
        """Evaluates terminal nodes for backpropagation."""
        winner = env.info["winner"]
        if winner == -1:
            return 0.0
        return 1.0 if winner == player else -1.0

    def expand(self) -> None:
        """Create lazy child stubs for all valid actions (no env cloning yet).

        Children are stored as unevaluated stubs that materialize their env only
        when selected during tree traversal.
        """
        if self.children or self.env.done:
            return
        for action in self.env.get_valid_actions():
            self.children[action] = AZNode(
                parent=self,
                parent_action=action,
                _source_env=self._env,
                _source_action=action,
            )

    def select_child(self, cpuct: float) -> AZNode:
        """Select the child node with the highest PUCT score."""
        sqrt_parent = math.sqrt(self.n_visits + 1)
        cpuct_sqrt = cpuct * sqrt_parent
        best_child = None
        best_score = -1e9
        for child in self.children.values():
            score = -child.avg_value + cpuct_sqrt * child.prob / (child.n_visits + 1)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def backpropagate(self, value: float) -> None:
        """Update the current node and its ancestors with the given value."""
        self.total_score += value
        self.n_visits += 1
        if self.parent is not None:
            self.parent.backpropagate(-value)

    def is_leaf(self) -> bool:
        """Check if the node is a leaf (no children)."""
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        """Check if the node represents a terminal state."""
        return self.env.done

    def __str__(self) -> str:
        return (
            f"Prob: {self.prob}\nTo play: {self.to_play}"
            f"\nChildren: {len(self.children)}\nVisits: {self.n_visits}"
            f"\nAvg value: {self.avg_value:.4f}"
        )


class AlphaZeroMCTS:
    """AlphaZero MCTS with lazy expansion and tree reuse between moves.

    Key optimizations vs a naive implementation:
    - Lazy env cloning: child environments are cloned only when first visited,
      reducing O(branching_factor x n_sims) clones to O(n_sims).
    - Tree reuse: call advance_root(action) after each move so subsequent
      searches start from a warm subtree rather than a fresh root.
    - Vectorized PUCT: child scores computed with numpy instead of a Python loop.
    """

    def __init__(
        self,
        net: AlphaZeroNet,
        n_simulations: int,
        cpuct: float,
        dirichlet_alpha: float = 1.0,
        dirichlet_eps: float | None = None,
    ):
        self.net = net
        self.n_simulations = n_simulations
        self.cpuct = cpuct
        self.skip_dirichlet = dirichlet_alpha is None or dirichlet_eps is None
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self._root: AZNode | None = None

    # ------------------------------------------------------------------
    # Root helpers
    # ------------------------------------------------------------------

    def _build_root(self, env: GenericEnv) -> AZNode:
        """Create root node, expand immediately, and inject Dirichlet noise."""
        root_env = env.clone()
        root = AZNode(env=root_env)
        valid_actions = root_env.get_valid_actions()
        policy, _ = self.net.predict(root_env.get_state())
        raw = np.array([policy[a - 1] for a in valid_actions], dtype=np.float32)
        if not self.skip_dirichlet:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_actions))
            raw = (1.0 - self.dirichlet_eps) * raw + self.dirichlet_eps * noise
        raw /= raw.sum()
        for action, prob in zip(valid_actions, raw):
            child = AZNode(parent=root, parent_action=action, _source_env=root_env, _source_action=action)
            child.prob = float(prob)
            root.children[action] = child
        return root

    def _apply_dirichlet(self, root: AZNode) -> None:
        """Re-apply Dirichlet noise to root's children after tree reuse."""
        if self.skip_dirichlet or not root.children:
            return
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, n in zip(actions, noise):
            child = root.children[action]
            child.prob = float((1.0 - self.dirichlet_eps) * child.prob + self.dirichlet_eps * n)

    # ------------------------------------------------------------------
    # Tree reuse
    # ------------------------------------------------------------------

    def advance_root(self, action: int) -> None:
        """Move the cached root to the subtree for *action* (tree reuse).

        Call this immediately after each game move during self-play so the next
        call to run() starts from an already-explored subtree.
        """
        if self._root is None or action not in self._root.children:
            self._root = None
            return
        new_root = self._root.children[action]
        new_root.parent = None
        new_root.parent_action = None
        _ = new_root.env  # materialize if still lazy
        self._root = new_root
        self._apply_dirichlet(self._root)

    def reset(self) -> None:
        """Discard the cached root. Call between independent games."""
        self._root = None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def run(self, env: GenericEnv, temperature: float = 0.0) -> tuple[int, AZNode]:
        """Run MCTS simulations and return (chosen_action, root).

        Args:
            env: Current game environment (not mutated).
            temperature: Action selection temperature. 0.0 = greedy.

        Returns:
            Tuple of (selected action, root AZNode with visit statistics).
        """
        self.net.eval()

        root = self._root if self._root is not None else self._build_root(env)

        for _ in range(self.n_simulations):
            node = root

            # SELECTION
            while not node.is_terminal() and not node.is_leaf():
                node = node.select_child(self.cpuct)

            # EXPANSION + EVALUATION
            if not node.is_terminal():
                node.expand()
                policy, value = self.net.predict(node.env.get_state())
                valid_actions = node.env.get_valid_actions()
                raw = np.array([policy[a - 1] for a in valid_actions], dtype=np.float32)
                raw /= raw.sum()
                for action, prob in zip(valid_actions, raw):
                    node.children[action].prob = float(prob)
                value = float(value)
            else:
                value = node.terminal_node_eval(node.env, node.to_play)

            # BACKPROPAGATION
            node.backpropagate(value)

        self._root = root
        return self.select_action(root, temperature), root

    def select_action(self, root: AZNode, temperature: float = 0.0) -> int:
        """Select an action from the root based on visit counts and temperature.

        Args:
            root: Root node after simulations.
            temperature: 0.0 = greedy (max visits), inf = uniform random.
        """
        actions = list(root.children.keys())
        counts = np.array([root.children[a].n_visits for a in actions], dtype=np.float32)
        if temperature is None or temperature == 0.0:
            return actions[int(np.argmax(counts))]
        elif temperature == np.inf:
            return np.random.choice(actions)
        else:
            counts = counts ** (1.0 / temperature)
            counts /= counts.sum()
            return np.random.choice(actions, p=counts)
