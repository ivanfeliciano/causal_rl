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
number_of_experiments = args.experiments if args.experiments > 0 else (1 << num) - 1
horizon = num
stochastic = args.stochastic
pmod = args.pmod
env_config = "stochastic" if stochastic else "deterministic"
env = LightEnv(horizon=horizon, num=num, structure=structure)

rewards = np.array([[None for _ in range(number_of_experiments)] for _ in range(4)])
for config in range(number_of_experiments):
    env.keep_struct = False
    env.reset()
    env.keep_struct = True
    print("Running training on new struct and new goal. {} / {}".format(config + 1, number_of_experiments))
    full_adj_list = aj_to_adj_list(env.aj)
    if pmod == "low":
        incomplete_adj_list = del_edges(deepcopy(full_adj_list), 0.25)
        incorrect_adj_list = shuffle_aj_mat(deepcopy(full_adj_list), 0.25)
    if pmod == "medium":
        incomplete_adj_list = del_edges(deepcopy(full_adj_list), 0.5)
        incorrect_adj_list = shuffle_aj_mat(deepcopy(full_adj_list), 0.5)
    if pmod == "high":
        incomplete_adj_list = del_edges(deepcopy(full_adj_list), 0.75)
        incorrect_adj_list = shuffle_aj_mat(deepcopy(full_adj_list), 0.75)
    print("finished adj mat initialization")
    vanilla_env = LightAndSwitchEnv(copy(env), full_adj_list, stochastic=stochastic)
    full_info_environment = LightAndSwitchEnv(copy(env), full_adj_list, stochastic=stochastic)
    partial_info_environment = LightAndSwitchEnv(copy(env), incomplete_adj_list, stochastic=stochastic)
    wrong_info_environment = LightAndSwitchEnv(copy(env), incorrect_adj_list, stochastic=stochastic)


    eps_partition = num * episodes * partition // 100
    eps_policy = EpsilonGreedy(1, 0.1, 0.1, eps_partition, full_info_environment)
    causal_eps_policy = EpsilonGreedy(1, 0.1, 0.1, eps_partition, full_info_environment, True)
    partial_causal_eps_policy = EpsilonGreedy(1, 0.1, 0.1, eps_partition, partial_info_environment, True)
    wrong_causal_eps_policy = EpsilonGreedy(1, 0.1, 0.1, eps_partition, wrong_info_environment, True)
    if draw:
        dir_name = "./drawings/{}_{}_{}_{}".format(structure, num, pmod, env_config)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        full_info_environment.causal_structure.draw_graph("{}/full_{}".format(dir_name, config)) 
        partial_info_environment.causal_structure.draw_graph("{}/partial_{}".format(dir_name, config)) 
        wrong_info_environment.causal_structure.draw_graph("{}/wrong_{}".format(dir_name, config))

    vanilla_q_learning = QLearningAgent(vanilla_env, eps_policy, episodes=episodes, mod_episode=mod_episode)
    causal_q_learning = QLearningAgent(full_info_environment, causal_eps_policy, episodes=episodes, mod_episode=mod_episode)
    partial_causal_q_learning = QLearningAgent(partial_info_environment, partial_causal_eps_policy, episodes=episodes, mod_episode=mod_episode)
    wrong_causal_q_learning = QLearningAgent(wrong_info_environment, wrong_causal_eps_policy, episodes=episodes, mod_episode=mod_episode)

    t_start = time.time()
    print("Running vanilla_q_learning")
    rewards[0][config] = np.array(vanilla_q_learning.train())
    print("{:.2f} seconds training Q-learning".format(time.time() - t_start))
    t_start = time.time()
    print("Running causal_q_learning")
    rewards[1][config] = np.array(causal_q_learning.train())
    print("{:.2f} seconds training Q-learning fully informed".format(time.time() - t_start))
    t_start = time.time()
    print("Running partial_causal_q_learning")
    rewards[2][config] = np.array(partial_causal_q_learning.train())
    print("{:.2f} seconds training Q-learning partially informed".format(time.time() - t_start))
    t_start = time.time()
    print("Running wrong_causal_q_learning")
    rewards[3][config] = np.array(wrong_causal_q_learning.train())
    print("{:.2f} seconds training Q-learning wrong informed".format(time.time() - t_start))


mean_vectors = [_ for _ in range(4)]
std_dev_vectors = [_ for _ in range(4)]
labels = ["Q-learning", "Q-learning + estructura completa", \
            "Q-learning + estructura parcial", "Q-learning + estructura incorrecta"]
for i in range(4):
    mean_vectors[i] = np.mean(rewards[i], axis=0)
    std_dev_vectors[i] = np.std(rewards[i], axis=0)

np.save("./rewards_data/latest_experiments/{}_{}_{}_{}_{}_eps_partition_{}".format(number_of_experiments, num, structure, episodes, env_config, partition), rewards)

x_axis = mod_episode * (np.arange(len(mean_vectors[0])))
dir_name_plot = "plots/qlearning/fixedgoal/{}/{}/{}/{}".format(structure, env_config, num, pmod)

if not os.path.exists(dir_name_plot):
    os.makedirs(dir_name_plot)
plot_rewards(x_axis, mean_vectors, std_dev_vectors, labels,\
    "Average reward comparison {} {}".format(num, structure), "{}/comparison_{}_{}_{}_{}_{}_eps_partition_{}".format(dir_name_plot, number_of_experiments, num, structure, episodes,env_config, partition))
# a = vanilla_q_learning.test()
# b = causal_q_learning.test()
# c = partial_causal_q_learning.test()
# d = wrong_causal_q_learning.test()
