import math
import random


class MCTSNode:
    def __init__(self, env, parent=None, parent_action=None, player_just_moved=None):
        self.env = env
        self.parent = parent
        self.parent_action = parent_action

        # Player who made the move leading to this node
        self.player_just_moved = player_just_moved

        self.children = {}              # action -> MCTSNode
        self.untried_actions = env.get_valid_actions()

        self.total_visits = 0
        self.total_value = 0.0          # from player_just_moved's perspective

    @property
    def avg_value(self):
        return 0.0 if self.total_visits == 0 else self.total_value / self.total_visits

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, cpuct):
        """
        Select child maximizing UCT from the perspective of
        the player to move at *this* node.
        """
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
    def __init__(self, n_simulations=1000, cpuct=1.4):
        self.n_simulations = n_simulations
        self.cpuct = cpuct

    def run(self, env):
        # Root player MUST be reset every move
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

            # -------- SELECTION --------
            while (
                not sim_env.done
                and node.is_fully_expanded()
                and node.children
            ):
                action, node = node.best_child(self.cpuct)
                sim_env.step(action)

            # -------- EXPANSION --------
            if not sim_env.done and node.untried_actions:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)

                sim_env.step(action)
                player_just_moved = 1 - sim_env.current_player

                child = MCTSNode(
                    sim_env.clone(),
                    parent=node,
                    parent_action=action,
                    player_just_moved=player_just_moved,
                )

                node.children[action] = child
                node = child

            # -------- SIMULATION --------
            result = self.rollout(sim_env, root_player)

            # -------- BACKPROP --------
            self.backpropagate(node, result, root_player)

        # Choose most visited action
        action, _ = max(
            root.children.items(),
            key=lambda item: item[1].total_visits,
        )
        return action

    def rollout(self, env, root_player):
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

    def backpropagate(self, node, result, root_player):
        """
        result is from root_player's perspective.
        Convert to each node's player_just_moved perspective.
        """
        while node is not None:
            node.total_visits += 1

            if node.player_just_moved == root_player:
                node.total_value += result
            else:
                node.total_value -= result

            node = node.parent
