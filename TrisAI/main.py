from env import TrisEnv
from agents import RandomAgent, Agent, HumanAgent
import numpy as np
import pickle
import pandas as pd

#define agents
ag1, ag2 = Agent(), Agent()

#load q-table
with open('qtable_v3.pickle', 'rb') as f:
  q_table = pickle.load(f)
ag1.load_q_table(q_table)
ag2.load_q_table(q_table)

#set to evaluation mode
ag1.eval()
ag2.eval()

env = TrisEnv(agents=[ag1, ag2], render_mode=None)

#play some games
n_games = 10000
print(f"Playing {n_games} games between the agents...")
cum_results = []
for i in range(n_games):
  state, done, info = env.reset()
  while not done:
    action = env.query_raw_action()
    state, done, info, traj, acts = env.step(action)
    if done:
      if info[env.player_ids[0]]["result"] == "win":
        cum_results.append(1)
        print(cum_results[-1])
      elif info[env.player_ids[0]]["result"] == "loss":
        cum_results.append(-1)
      elif info[env.player_ids[0]]["result"] == "draw":
        cum_results.append(0)
      
print("Testing results: (1 = ag1 win, -1 = ag1 loss, 0 = draw)")
print(pd.Series(cum_results).value_counts())

#play some games against a random opponent
env = TrisEnv(agents=[ag1, RandomAgent()], render_mode=None)

print(f"Playing {n_games} games against a random agent...")
cum_results = []
for i in range(n_games):
  state, done, info = env.reset()
  while not done:
    action = env.query_raw_action()
    state, done, info, traj, acts = env.step(action)
    if done:
      if info[env.player_ids[0]]["result"] == "win":
        cum_results.append(1)
      elif info[env.player_ids[0]]["result"] == "loss":
        cum_results.append(-1)
        #uncomment to print games where the agent lost against random
        #for i in traj[1]:
        #  env.render_mode = "text"
        #  env.board = i
        #  env.render()
        #  env.render_mode = None
      elif info[env.player_ids[0]]["result"] == "draw":
        cum_results.append(0)
print("Testing results: (1 = ag1 win, -1 = ag1 loss, 0 = draw)")
print(pd.Series(cum_results).value_counts())


#game with human
env = TrisEnv(agents=[HumanAgent(), ag1], render_mode="text")

while int(input("Play a game with the agent (1/0)? ")) != 0:
  state, done, info = env.reset()
  while not done:
    action = env.query_raw_action()
    state, done, info, traj, acts = env.step(action)
