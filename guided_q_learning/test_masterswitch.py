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


parser = argparse.ArgumentParser(description='Run Q-learning and Q-learning CM light switch problem')

parser.add_argument("--stochastic", help="change to simple stochastic enviroments (0.8 prob of do the choosen action)",\
                    action="store_true")
parser.add_argument("--episodes", type=int, default=1000, help="# of episodes per experiment")
parser.add_argument("--basedir", type=str, default="/home/ivan/Documentos/causal_rl/guided_q_learning/results_thesis/light_switches", help="path to save results")
parser.add_argument("--partition", type=int, default=100, help="at what percentage begins with small epsilon")
parser.add_argument("--mod", type=int, default=50, help="# block of avg episodes")
parser.add_argument("--experiments", type=int, default=0, help="# of experiments, one experiment corresponds to one structure and one goal")
parser.add_argument("--num", type=int, default=5, help="Number of switches")
parser.add_argument("--structure", type=str, default="one_to_one", help="structure of the graph")
parser.add_argument("--pmod", type=str, default="low", help="percentage of incorrectness")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbosite activated")
parser.add_argument("-d", "--draw", action="store_true", help="Draw graphs activated")

args = parser.parse_args()

mod_episode = args.mod
num = args.num 
partition = args.partition
episodes = args.episodes
structure = args.structure
draw = args.draw
base_dir = args.basedir
number_of_experiments = args.experiments if args.experiments > 0 else (1 << num) - 1
stochastic = args.stochastic
pmod = args.pmod

horizon = num
env_config = "stochastic" if stochastic else "deterministic"
env = LightEnv(horizon=horizon, num=num, structure=structure)
if pmod == "low":
	pmod_factor = 0.25
if pmod == "medium":
	pmod_factor = 0.5
if pmod == "high":
	pmod_factor = 0.75
n_algorithms = 4

rewards = np.array([[None for _ in range(number_of_experiments)] for _ in range(n_algorithms)])
eps_partition = num * episodes * partition // 100

file_name = "{}_{}_{}_{}_N_{}_experiments_{}_episodes_{}_eps_{}".format(env_config, pmod, pmod_factor, structure, num, number_of_experiments, episodes, eps_partition)

for config in range(number_of_experiments):
    env.keep_struct = False
    env.reset()
    env.keep_struct = True
    print("Running training on new struct and new goal. {} / {}".format(config + 1, number_of_experiments))
    if env.structure == "masterswitch":
        full_adj_list = aj_to_adj_list(env, masterswitch=True)
    if env.structure == "masterswitch":
        full_adj_list = aj_to_adj_list(env, masterswitch=True)
    else:
        full_adj_list = aj_to_adj_list(env)
    incomplete_adj_list = del_edges(deepcopy(full_adj_list), pmod_factor)
    incorrect_adj_list = shuffle_aj_mat(deepcopy(full_adj_list), pmod_factor)
    print("finished adj mat initialization")
    vanilla_env = LightAndSwitchEnv(copy(env), full_adj_list, stochastic=stochastic)
    full_info_environment = LightAndSwitchEnv(copy(env), full_adj_list, stochastic=stochastic)
    partial_info_environment = LightAndSwitchEnv(copy(env), incomplete_adj_list, stochastic=stochastic)
    wrong_info_environment = LightAndSwitchEnv(copy(env), incorrect_adj_list, stochastic=stochastic)
    eps_policy = EpsilonGreedy(1, 0.1, 0.1, eps_partition, full_info_environment)
    causal_eps_policy = EpsilonGreedy(1, 0.1, 0.1, eps_partition, full_info_environment, True)
    partial_causal_eps_policy = EpsilonGreedy(1, 0.1, 0.1, eps_partition, partial_info_environment, True)
    wrong_causal_eps_policy = EpsilonGreedy(1, 0.1, 0.1, eps_partition, wrong_info_environment, True)
    
