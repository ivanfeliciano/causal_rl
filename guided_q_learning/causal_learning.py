from copy import copy, deepcopy

import numpy as np
import pandas as pd
import networkx as nx

from causality.inference.search import IC
from causality.inference.independence_tests import RobustRegressionTest

from env.light_env import LightEnv
from environments import LightAndSwitchEnv
from utils.lights_env_helper import aj_to_adj_list
from policy import EpsilonGreedy
from agents import QLearningAgent

env = LightEnv()

n = 5
episodes = 3
mod_episode = 1
env.keep_struct = False
env.reset()
env.keep_struct = True

full_adj_list = aj_to_adj_list(env.aj)
print(full_adj_list)
full_info_environment = LightAndSwitchEnv(env, full_adj_list)

# causal_eps_policy = EpsilonGreedy(1, 0.1, 0.1, n, full_info_environment, True)

full_info_environment.causal_structure.draw_graph("full_struct")


actions = []
states = []
episodes = 500
for episode in range(episodes):
	done = False
	full_info_environment.reset()
	while not done:
		state = full_info_environment.get_state()
		print(state)
		action = full_info_environment.sample_action()
		print(action)
		new_state, reward, done, info =  full_info_environment.step(action)
		one_hot = [0 for _ in range(n + 1)]
		one_hot[action] = 1
		actions.append(one_hot)
		states.append(full_info_environment.get_state())
print(actions)
print(states)
actions = np.array(actions)
states = np.array(states)
print(actions)
print(states)
table = np.hstack([actions, states])
df = pd.DataFrame(data=table, columns = ["a1", "a2", "a3", "a4", "a5", "Nada", "x1", "x2", "x3", "x4", "x5"])
print(df)
ic_algorithm = IC(RobustRegressionTest)
variable_types = { i : 'c' for i in df.columns} 
print(variable_types)
graph = ic_algorithm.search(df, variable_types)
print(graph.edges(data=True))
# causal_q_learning = QLearningAgent(full_info_environment, causal_eps_policy, episodes=episodes, mod_episode=mod_episode)

# r = np.array(causal_q_learning.train())

# print(r)