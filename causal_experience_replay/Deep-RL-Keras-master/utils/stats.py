import numpy as np
from random import random

def gather_stats(agent, env, stochastic=False):
  """ Compute average rewards over 10 episodes
  """
  score = []
  for k in range(10):
      old_state = env.reset()
      cumul_r, done = 0, False
      while not done:
          a = agent.policy_action(old_state)
          if stochastic and random() < 0.75:
              a = 0 if a == 0 else 0
          old_state, r, done, _ = env.step(a)
          cumul_r += r
      score.append(cumul_r)
  return np.mean(np.array(score)), np.std(np.array(score))
