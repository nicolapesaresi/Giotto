#in this file I fine tune the model by making it play against random agents,
#without rewarding it for a win but only penalizing it for a loss

from env import TrisEnv
from agents import RandomAgent, Agent, HumanAgent
import numpy as np
import pickle

reward_map = {
    "win": 0,
    "loss": -10,
    "draw": 0,
}

ag1 = Agent(learning_rate=0.5, discount_factor=0.9, exploration_rate=0.3)

with open('qtable.pickle', 'rb') as f:
  q_table = pickle.load(f)
ag1.load_q_table(q_table)

ag1.train()
#self play
env = TrisEnv(agents=[ag1, RandomAgent()], render_mode=None)

#training loop
cum_results = []
for episode in range(1000000):
  if episode % 100000 == 0:
    print(f"Episode {episode}")
  state, done, info = env.reset()
  rewards = {env.player_ids[0]: 0, env.player_ids[1]: 0}
  while not done:
    now_playing = env.game_pointer  #needed to save states to update q-values
    action = env.query_raw_action()
    state, done, info, trajectories, actions = env.step(action)
    #print(trajectories)
    #print(actions)
    if done:  #rewards only at the end of the game
      #update q-values with rewards or penalties in case of loss
      if env.info[env.player_ids[0]]["result"] == "win":
        cum_results.append(1)
      elif env.info[env.player_ids[0]]["result"] == "loss":
        cum_results.append(-1)
        if env.agents[1].mode == "train":
          rewards[1] = reward_map[env.info[1]["result"]]
          env.agents[1].update_q_values(trajectories[1], actions[1],
                                         rewards[1])
      else:
        cum_results.append(0)
        
      
      #print(cum_results)
      #print(info)
      #print(rewards)
      #print(trajectories)
      #print(ag1.q_table)

import pandas as pd

print("Training results:")
print(pd.Series(cum_results).value_counts())
print(len(cum_results))

#set to eval to test results
ag1.eval()
old_ag = Agent()
with open('qtable_backup.pickle', 'rb') as f:
  q_table = pickle.load(f)
old_ag.load_q_table(q_table)
old_ag.eval()

env = TrisEnv(agents=[ag1, old_ag], render_mode=None)
#play some games with previous version to evaluate the training
cum_results = []
for i in range(10000):
  state, done, info = env.reset()
  while not done:
    action = env.query_raw_action()
    state, done, info, traj, acts = env.step(action)
    if done:
      if info[env.player_ids[0]]["result"] == "win":
        cum_results.append(1)
      elif info[env.player_ids[0]]["result"] == "loss":
        cum_results.append(-1)
      elif info[env.player_ids[0]]["result"] == "draw":
        cum_results.append(0)
print("testing results:")
print(pd.Series(cum_results).value_counts())

#play some games with random opp to evaluate the training
env = TrisEnv(agents=[ag1, RandomAgent()], render_mode=None)
cum_results = []
for i in range(10000):
  state, done, info = env.reset()
  while not done:
    action = env.query_raw_action()
    state, done, info, traj, acts = env.step(action)
    if done:
      if info[env.player_ids[0]]["result"] == "win":
        cum_results.append(1)
      elif info[env.player_ids[0]]["result"] == "loss":
        cum_results.append(-1)
      elif info[env.player_ids[0]]["result"] == "draw":
        cum_results.append(0)
print("testing results:")
print(pd.Series(cum_results).value_counts())

#game with human
env = TrisEnv(agents=[HumanAgent(), ag1], render_mode="text")

#test with human play
while int(input("Play a game (1/0)? ")) != 0:
  state, done, info = env.reset()
  while not done:
    action = env.query_raw_action()
    state, done, info, traj, acts = env.step(action)

#print empty board q-values
print(ag1.q_table["000000000"])

#save trained q-table
with open('qtable_ft.pickle', 'wb') as f:
  pickle.dump(ag1.q_table, f)