import numpy as np
from agents import RandomAgent, Agent, HumanAgent

class TrisEnv():
  """Class for the game of Tic-Tac-Toe (Tris)
  """
  metadata = {
    "render_modes": ["text"],
  }
  
  def __init__(self, agents=[RandomAgent(), RandomAgent()], render_mode=None, seed=None) -> None:
    #check agents and render_mode are valid
    for agent in agents:
      if not (isinstance(agent, RandomAgent) or isinstance(agent, Agent) or isinstance(agent, HumanAgent)):
        raise ValueError("incorrect agent specification")
    self.agent_list = agents
        
    if not (render_mode in self.metadata["render_modes"] or render_mode == None):
      raise ValueError("render_mode not valid")
    self.render_mode = render_mode

    #set seed (for random agent and training) and initialize game
    np.random.seed(seed)
    
    self.player_ids = [1, 2]
    self.signs = {
      self.player_ids[0]: "X",
      self.player_ids[1]: "O",
    }
    self.agents = {
      self.player_ids[0]: self.agent_list[0],
      self.player_ids[1]: self.agent_list[1],
    }
  
  def reset(self) -> None:
    """Resets the game to the initial state
    
    Returns:
      self.state (np.ndarray): current state of the game
      self.done (bool): whether the game is over
      self.info (dict): informations about the game
    """
    self.board = np.full((3, 3), " ")
    self.state = np.zeros((3, 3), dtype=int)
    self.starting_id = np.random.choice(self.player_ids)
    #self.starting_id = 1
    self.game_pointer = self.starting_id

    self.n_moves = 0
    self.done = False

    self.trajectories = {
      self.player_ids[0]: [],
      self.player_ids[1]: [],
    }
    self.actions = {
      self.player_ids[0]: [],
      self.player_ids[1]: [],
    }
    self.info = {
      self.player_ids[0]: {"result":"in_progress"},
      self.player_ids[1]: {"result":"in_progress"},
    }
    
    return self.state, self.done, self.info

  def action_to_coords(self, action_raw):
    """Turns raw action (0-8) into tuple (row, col)
    
    Args:
      action_raw (int): raw action (0-8 int)

    Returns:
      (int, int): action tuple (row, col)
    """
    return (action_raw // 3, action_raw % 3)

  def check_win(self, sign):
    """Check if player has won
    
    Returns:
      (bool): True if player has won
    """
    if (self.board[0, 0] == self.board[1, 1] == self.board[2, 2] == sign
        or self.board[0, 2] == self.board[1, 1] == self.board[2, 0] == sign
        or self.board[0, 0] == self.board[0, 1] == self.board[0, 2] == sign
        or self.board[1, 0] == self.board[1, 1] == self.board[1, 2] == sign
        or self.board[2, 0] == self.board[2, 1] == self.board[2, 2] == sign
        or self.board[0, 0] == self.board[1, 0] == self.board[2, 0] == sign
        or self.board[0, 1] == self.board[1, 1] == self.board[2, 1] == sign
        or self.board[0, 2] == self.board[1, 2] == self.board[2, 2] == sign):
      return True
    else:
      return False

  def check_draw(self):
    """Check if game is a draw, if it's already not a win
    
    Returns:
      (bool): True if game is a draw
    """
    if self.n_moves >= 9:
      return True
    else:
      return False

  def get_legal_actions(self):
    """Get list of legal actions
    
    Returns:
      (np.ndarray): list of legal actions, expressed in raw integers (0-8)
    """
    flat_state = self.state.flatten()
    return np.where(flat_state == 0)[0]

  def query_raw_action(self):
    """Queries the currently playing agent for his next action
    
    Returns:
      (int): raw selected action (0-8)
    """
    current_agent = self.agents[self.game_pointer]
    return current_agent.select_raw_action(self.state, self.get_legal_actions())
    
  def step(self, action_raw: int):
    """Plays out a player's turn, by playing his action and updating the game state.
    Then checks if the game is over.
    
    Args:
      action_raw (int): raw action of the current player (0-8)

    Returns:
      self.state (np.ndarray): state of the game after the action
      self.done (bool): whether the game is over
      self.info (dict): informations about the game
      self.trajectories (dict): list of states visited in the game
      self.actions (dict): list of actions taken in the game
    """
    self.trajectories[self.game_pointer].append(self.state.copy())
    
    action = self.action_to_coords(action_raw)
    self.state[action] = self.game_pointer
    self.board[action] = self.signs[self.game_pointer]
    self.actions[self.game_pointer].append(action_raw)
    self.n_moves += 1

    if self.check_win(self.signs[self.game_pointer]):
      self.done = True
      self.info[self.game_pointer]["result"] = "win"
      self.info[(self.game_pointer % len(self.player_ids)) + 1]["result"] = "loss"
      #append final state to trajectory
      for id in self.player_ids:
        self.trajectories[id].append(self.state.copy())
    elif self.check_draw():
      self.done = True
      self.info[self.game_pointer]["result"] = "draw"
      self.info[(self.game_pointer % len(self.player_ids)) + 1]["result"] = "draw"
      #append final state to trajectory
      for id in self.player_ids:
        self.trajectories[id].append(self.state.copy())

    self.render()
    
    self.game_pointer = (self.game_pointer % len(self.player_ids)) + 1

    return self.state, self.done, self.info, self.trajectories, self.actions

  def render(self):
    """Renders the game state in the specified render mode
    """
    if self.render_mode == "text":
      print(f"Move {self.n_moves}. {self.signs[self.starting_id]} started the game.")
      print(f" {self.board[0,0]} | {self.board[0,1]} | {self.board[0,2]} ")
      print("-----------")
      print(f" {self.board[1,0]} | {self.board[1,1]} | {self.board[1,2]} ")
      print("-----------")
      print(f" {self.board[2,0]} | {self.board[2,1]} | {self.board[2,2]} ")

      if self.info[self.player_ids[0]]["result"] == "draw":
        print("Game is a tie.")
      elif self.info[self.player_ids[0]]["result"] == "win":
        print(f"{self.signs[self.player_ids[0]]} won.")
      elif self.info[self.player_ids[1]]["result"] == "win":
        print(f"{self.signs[self.player_ids[1]]} won.")