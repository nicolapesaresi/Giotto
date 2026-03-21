import random

from giotto.envs.connect4 import Connect4Env
from giotto.envs.tris import TrisEnv


def random_board_tris():
    """Generates random tic tac toe board position."""
    generated = False

    while not generated:
        env = TrisEnv()
        env.reset()  # random starting player
        n_moves = random.randint(1, 8)  # no empty or full boards
        for _ in range(n_moves):
            if not env.done:
                valid_actions = env.get_valid_actions()
                env.step(random.choice(valid_actions))
        if not env.done:
            generated = True
    # env.render()
    return env.get_state(), env


def random_board_connect4():
    """Generates random connect4 board position."""
    generated = False

    while not generated:
        env = Connect4Env()
        env.reset()  # random starting player
        n_moves = random.randint(1, 41)  # no empty or full boards
        for _ in range(n_moves):
            if not env.done:
                valid_actions = env.get_valid_actions()
                env.step(random.choice(valid_actions))
        if not env.done:
            generated = True
    # env.render()
    return env.get_state(), env
