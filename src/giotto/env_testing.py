  from env import TrisEnv
  from agents import RandomAgent, Agent, HumanAgent

  env = TrisEnv(agents=[HumanAgent(), RandomAgent()], render_mode="text")

  state, done, info = env.reset()
  while not done:
    action = env.query_raw_action()
    state, done, info = env.step(action)