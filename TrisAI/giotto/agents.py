import numpy as np


class RandomAgent():
  """Agent that plays random moves
  """
  def __init__(self) -> None:
    self.name = "RandomAgent"

  def select_raw_action(self, state, legal_actions):
    """Selects a random action from the legal actions

    Args:
      state (np.ndarray): current state of the game
      legal_actions (np.ndarray): list of legal actions

    Returns:
      int: random action
    """
    return np.random.choice(legal_actions)


class HumanAgent():
  """Agent that plays an human-inputed move
  """
  def __init__(self) -> None:
    self.name = "HumanAgent"

  def select_raw_action(self, state, legal_actions):
    """Queries human for an actions and, if it's legal, returns it
    
    Args:
      state (np.ndarray): current state of the game
      lega_actions (np.ndarray): list of legal actions

    Returns:
      int: human-inputed action
    """
    valid = False
    while not valid:
      raw_action = int(input("Select an action (1-9): ")) - 1
      if raw_action in legal_actions:
        valid = True
      else:
        print(f"Illegal action. Valid actions: {legal_actions + 1}")

    return raw_action


class Agent():
  """Q-Learning agent
  """
  def __init__(self, learning_rate=0.01, discount_factor=0.9, exploration_rate = 0.1) -> None:
    self.name = "Agent"
    self.max_actions = 9
    self.set_params(learning_rate, discount_factor, exploration_rate)
    self.q_table = {}

    self.mode = "train"

  def set_params(self, learning_rate, discount_factor, exploration_rate):
    """Sets model parameters
    
    Args:
      learning_rate (float): learning rate for the model
      discount_factor (float): discount factor for backpropagation of rewards to previous states
      exploration_rate (float): exploration rate for the model (% of random actions in training)
    """
    self.learning_rate = learning_rate
    self.discount_factor = discount_factor
    self.exploration_rate = exploration_rate

  def load_q_table(self, q_table):
    """Loads q-table from a file
    
    Args:
      q_table (dict): q-table to be loaded
    """
    self.q_table = q_table
  
  def train(self):
    """Sets the agent in training mode, meaning it will explore
    """
    self.mode = "train"
    
  def eval(self):
    """Sets the agent in playing mode, so it will always take the best action
    """
    self.mode = "eval"

  def get_state_id(self, state):
    """Encodes state in a string that can serve as key of the q-values dict
    
    Args:
      state (np.ndarray): current state of the game

    Returns:
      state_id (str): string id of the state
    """
    state_id = state.flatten().astype(str)
    state_id = ''.join(state_id)
    return state_id
    
  def get_q_values(self, state_id):
    """Retrieves current q-values for a state from the q-table
    
    Args:
      state_id (str): string id of the state

    Returns:
      (np.ndarray): q-values for the state
    """
    if state_id not in self.q_table:
      self.q_table[state_id] = np.zeros(self.max_actions)
    return self.q_table[state_id]
  
  def select_raw_action(self, state, legal_actions):
    """Selects an action based on the current state and legal actions.
    
    first randomly sample a number between 0 and 1
    if it's lower than the exploration rate, select a random action
    else play the best action according to q-values

    if model is set in eval mode, will always play best action

    Args:
      state (np.ndarray): current state of the game
      legal_actions (np.ndarray): list of legal actions

    Returns:
      int: selected action
    """
    if self.mode == "train" and np.random.uniform() < self.exploration_rate: #explore
      return np.random.choice(legal_actions) #random action
    else: #exploit
      state_id = self.get_state_id(state)
      q_values = self.get_q_values(state_id)
      legal_q_values = q_values[legal_actions]
      best_action_idx = np.argmax(legal_q_values)
      action = legal_actions[best_action_idx]
      return action

  
  def update_q_values(self, trajectory, actions, reward):
    """Updates the q-values for a trajectory based on the final reward and the final state
    
    Args: 
      trajectory (list): list of states visited in the game
      actions (list): list of actions taken in the game
      reward (float): reward for the final state
    """
    if self.mode == "train":
      backward_trajectory = trajectory[::-1]
      backward_actions = actions[::-1]
      for i in range(len(backward_trajectory) - 1):
        old_state = backward_trajectory[i + 1]
        new_state = backward_trajectory[i]
        action = backward_actions[i]
        old_state_id = self.get_state_id(old_state)
        old_q_values = self.get_q_values(old_state_id)
        new_state_id = self.get_state_id(new_state)
        new_q_values = self.get_q_values(new_state_id)
        old_q_values[action] += self.learning_rate * (reward + self.discount_factor * np.max(new_q_values) - old_q_values[action])
        
        #not necessary? list updates automatically
        #self.q_table[old_state_id] = old_q_values


  