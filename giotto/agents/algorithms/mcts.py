from __future__ import annotations
from giotto.envs.generic import GenericEnv
import math
import random


class MCTSNode:
    """Node in MCTS tree."""

    def __init__(
        self,
        env: GenericEnv,
        parent: MCTSNode = None,
        parent_action: int = None,
        player_just_moved: int = None,
    ):
        """Instantiates MCTS node.
        Args:
            env: environment state at this node.
            parent: parent node.
            parent_action: action taken to reach this node from parent.
            player_just_moved: index of player who made the move leading to this node.
        """
        self.env = env
        self.parent = parent
        self.parent_action = parent_action

        # Player who made the move leading to this node
        self.player_just_moved = player_just_moved

        self.children: dict[int, MCTSNode] = {}
        self.untried_actions = env.get_valid_actions()

        self.total_visits = 0
        self.total_value = 0.0  # from player_just_moved's perspective

    @property
    def avg_value(self):
        """Returns average value of this node."""
        return 0.0 if self.total_visits == 0 else self.total_value / self.total_visits

    def is_fully_expanded(self):
        """Determines if all actions have been tried from this node."""
        return len(self.untried_actions) == 0

    def best_child(self, cpuct: float):
        """Selects child maximizing UCT from the perspective of the player to move at this node."""
        best_score = -math.inf
        best_children = []

        for action, child in self.children.items():
            # Player to move at this node
            player_to_move = 1 - self.player_just_moved

            # child.avg_value is from child.player_just_moved's perspective
            if child.player_just_moved == player_to_move:
                exploitation = child.avg_value
            else:
                exploitation = -child.avg_value

            exploration = cpuct * math.sqrt(
                math.log(self.total_visits) / child.total_visits
            )

            ucb = exploitation + exploration

            if ucb > best_score:
                best_score = ucb
                best_children = [(action, child)]
            elif math.isclose(ucb, best_score):
                best_children.append((action, child))

        return random.choice(best_children)


class MCTS:
    """Monte Carlo Tree Search algorithm."""

    def __init__(self, n_simulations: int = 1000, cpuct: float = 1.4):
        """Instantiates MCTS class.
        Args:
            n_simulations: number of simulations to run per move.
            cpuct: exploration constant.
        """
        self.n_simulations = n_simulations
        self.cpuct = cpuct

    def run(self, env: GenericEnv):
        """Runs MCTS to select action."""
        root_player = env.current_player

        root_env = env.clone()
        root = MCTSNode(
            root_env,
            parent=None,
            parent_action=None,
            player_just_moved=1 - root_env.current_player,
        )

        for _ in range(self.n_simulations):
            node = root
            sim_env = env.clone()

            # SELECTION
            while not sim_env.done and node.is_fully_expanded() and node.children:
                action, node = node.best_child(self.cpuct)
                sim_env.step(action)

            # EXPANSION
            if not sim_env.done and node.untried_actions:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)

                player_just_moved = sim_env.current_player  # putting this after step is wrong because current_player isn't uppdated by env after terminal move
                sim_env.step(action)

                child = MCTSNode(
                    sim_env.clone(),
                    parent=node,
                    parent_action=action,
                    player_just_moved=player_just_moved,
                )

                node.children[action] = child
                node = child

            # SIMULATION
            result = self.rollout(sim_env, root_player)

            # BACKPROPAGATION
            self.backpropagate(node, result, root_player)

        # choose most visited action
        action, _ = max(
            root.children.items(),
            key=lambda item: item[1].total_visits,
        )
        return action

    def rollout(self, env: GenericEnv, root_player: int):
        """Random playout until terminal state."""
        while not env.done:
            env.step(random.choice(env.get_valid_actions()))

        winner = env.info["winner"]
        if winner == -1:
            return 0
        elif winner == root_player:
            return 1
        else:
            return -1

    def backpropagate(self, node: MCTSNode, result: int, root_player: int):
        """Backpropagates simulation result up the tree."""
        while node is not None:
            node.total_visits += 1

            if node.player_just_moved == root_player:
                node.total_value += result
            else:
                node.total_value -= result

            node = node.parent
