from tqdm import tqdm
from giotto.envs.generic import GenericEnv
from giotto.agents.generic import GenericAgent
from collections import Counter
import random

def play_game(env: GenericEnv, agents: list[GenericAgent], starter:int=0, render:bool = True):
    """Plays a game between two agents.
    Args:
        env: game environment.
        agents: list of two agents.
        starter: index of player that has to start.
        render: wether to print the game.
    Returns:
        winner_sign, winner_idx, winner_name
    """
    env.reset(starting_player=starter)

    while not env.done:
        if render:
            env.render()
        action = agents[env.current_player].select_action(env)
        env.step(action)
    
    if render:
        env.render() # final position
        if env.info["winner"] == -1:
            print("Game over - Draw.")
        else:   
            print(f"Game over - {env.signs[env.info['winner']]} wins.")
    winner_sign = env.signs[env.info['winner']]
    winner_idx = env.info["winner"]
    winner_name = agents[env.info['winner']].name if env.info['winner'] != -1 else "Draw"
    return winner_sign, winner_idx, winner_name


def play_n_games(n_games: int, env: GenericEnv, agents: list[GenericAgent], invert_starts: bool = True):
    """Plays n games between two agents and returns stats."""
    winners = Counter({agents[0].name: 0, agents[1].name: 0, "Draw": 0})
    pbar = tqdm(range(n_games), desc="Playing games", ncols=100, leave=True)
    
    starter = 0
    for _ in pbar:
        new_env = env.clone()
        _, _, winner = play_game(new_env, agents, starter, render=False)
        winners[winner] += 1

        if invert_starts:
            starter = (starter + 1) % 2

        pbar.set_postfix({
            agents[0].name: winners[agents[0].name],
            agents[1].name: winners[agents[1].name],
            "Draw": winners["Draw"]
        })

    # Final recap
    for agent in agents:
        print(f"{agent.name}: {winners[agent.name]} / {n_games} ({(winners[agent.name]/n_games)*100:.2f}%)")
    print(f"Draws: {winners['Draw']} / {n_games} ({(winners['Draw']/n_games)*100:.2f}%)")


def initialized_game(env: GenericEnv, agents: list[GenericAgent], starter:int=0, first_move:int|None = None, render:bool = True):
    """Plays a game between two agents with fixed or random first move.
    Args:
        env: game environment.
        agents: list of two agents.
        starter: index of player that has to start.
        firstmove: first move of the game. If None, chosen at random.
        render: wether to print the game.
    Returns:
        winner_sign, winner_idx, winner_name
    """
    env.reset(starting_player=starter)

    if first_move is None:
        first_move = random.choice(env.get_valid_actions())

    if render:
        env.render()
    env.step(first_move)

    while not env.done:
        if render:
            env.render()
        action = agents[env.current_player].select_action(env)
        env.step(action)
    
    if render:
        env.render() # final position
        if env.info["winner"] == -1:
            print("Game over - Draw.")
        else:   
            print(f"Game over - {env.signs[env.info['winner']]} wins.")
    winner_sign = env.signs[env.info['winner']]
    winner_idx = env.info["winner"]
    winner_name = agents[env.info['winner']].name if env.info['winner'] != -1 else "Draw"
    return winner_sign, winner_idx, winner_name