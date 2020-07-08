import time
import os
import argparse
from copy import deepcopy, copy

import numpy as np
import matplotlib.pyplot as plt

from env.light_env import LightEnv
from environments import LightAndSwitchEnv
from agents import QLearningAgent
from utils.lights_env_helper import aj_to_adj_list, remove_edges, to_wrong_graph, del_edges, shuffle_aj_mat
from utils.vis_utils import plot_rewards
from policy import EpsilonGreedy

num = 5
horizon = num
structure = "masterswitch"
env = LightEnv(horizon=horizon, num=num, structure=structure)

env.keep_struct = False
env.reset()
env.keep_struct = True
print(env.aj)
print(env._get_obs()[0])


num = 5
horizon = num
structure = "one_to_one"
env = LightEnv(horizon=horizon, num=num, structure=structure)

env.keep_struct = False
env.reset()
env.keep_struct = True
print(env.aj)
print(env._get_obs()[0])
