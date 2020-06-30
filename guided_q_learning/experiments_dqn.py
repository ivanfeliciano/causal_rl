import time
import os
import argparse
from copy import deepcopy, copy

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from rl.policy import LinearAnnealedPolicy

from env.light_env import LightEnv
from environments import LightAndSwitchEnv
from agents import DQN
from utils.lights_env_helper import aj_to_adj_list, remove_edges, to_wrong_graph, del_edges, shuffle_aj_mat
from utils.vis_utils import plot_rewards
from policy import EpsilonGreedyDQN
from processor import LightEnvProcessor

parser = argparse.ArgumentParser(description='Run DQQN and DQN CM light switch problem')
parser.add_argument("--stochastic", help="change to simple stochastic enviroments (0.8 prob of do the choosen action)",\
                    action="store_true")
parser.add_argument("--episodes", type=int, default=1000, help="# of episodes per experiment")
parser.add_argument("--mod", type=int, default=50, help="# block of avg episodes")
parser.add_argument("--experiments", type=int, default=0, help="# of experiments, one experiment corresponds to one structure and one goal")
parser.add_argument("--num", type=int, default=5, help="Number of switches")
parser.add_argument("--structure", type=str, default="one_to_one", help="structure of the graph")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbosite activated")
parser.add_argument("-d", "--draw", action="store_true", help="Draw graphs activated")
parser.add_argument("--pmod", type=str, default="low", help="percentage of incorrectness")
parser.add_argument("--partition", type=int, default=100, help="at what percentage begins with small epsilon")
args = parser.parse_args()
mod_episode = args.mod
partition = args.partition
num = args.num 
episodes = args.episodes
structure = args.structure
draw = args.draw
number_of_experiments = args.experiments if args.experiments > 0 else (1 << num) - 1
horizon = num
stochastic = args.stochastic
pmod = args.pmod

model = load_model('models/multilabel_classifier84.h5')

env = LightEnv(horizon=horizon, num=num, structure=structure)
rewards = np.array([[None for _ in range(number_of_experiments)] for _ in range(4)])
processor = LightEnvProcessor()
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
    
    print("finished adj mat init")
    vanilla_env = LightAndSwitchEnv(copy(env), full_adj_list, discrete=False)
    full_info_environment = LightAndSwitchEnv(copy(env), full_adj_list, discrete=False)
    partial_info_environment = LightAndSwitchEnv(copy(env), incomplete_adj_list, discrete=False)
    wrong_info_environment = LightAndSwitchEnv(copy(env), incorrect_adj_list, discrete=False)

    nb_steps = num * episodes * partition // 100
    eps_policy = LinearAnnealedPolicy(EpsilonGreedyDQN(vanilla_env, num, None, False),\
                                            attr='eps', value_max=1., value_min=.1,\
                                            value_test=.05, nb_steps=nb_steps)
    causal_eps_policy = LinearAnnealedPolicy(EpsilonGreedyDQN(full_info_environment, num, model, True),\
                                            attr='eps', value_max=1., value_min=.1,\
                                            value_test=.05, nb_steps=nb_steps)
    partial_causal_eps_policy = LinearAnnealedPolicy(EpsilonGreedyDQN(partial_info_environment, num, model, True),\
                                            attr='eps', value_max=1., value_min=.1,\
                                            value_test=.05, nb_steps=nb_steps)
    wrong_causal_eps_policy = LinearAnnealedPolicy(EpsilonGreedyDQN(wrong_info_environment, num, model, True),\
                                            attr='eps', value_max=1., value_min=.1,\
                                            value_test=.05, nb_steps=nb_steps)                                
    if draw:
        dir_name = "./drawings/{}_{}_{}".format(structure, num, "sto" if stochastic else "det")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        full_info_environment.causal_structure.draw_graph("{}/dqn_full_{}".format(dir_name, config)) 
        partial_info_environment.causal_structure.draw_graph("{}/dqn_partial_{}".format(dir_name, config)) 
        wrong_info_environment.causal_structure.draw_graph("{}/dqn_wrong_{}".format(dir_name, config))


    vanilla_q_learning = DQN(vanilla_env, num + 1, processor, eps_policy, episodes, mod_episode=mod_episode)
    causal_q_learning = DQN(full_info_environment, num + 1, processor, causal_eps_policy, episodes, mod_episode=mod_episode)
    partial_causal_q_learning = DQN(partial_info_environment, num + 1, processor, partial_causal_eps_policy, episodes, mod_episode=mod_episode)
    wrong_causal_q_learning = DQN(wrong_info_environment, num + 1, processor, wrong_causal_eps_policy, episodes, mod_episode=mod_episode)
    # t_start = time.time()
    rewards[0][config] = np.array(vanilla_q_learning.train())
    del vanilla_q_learning
    # print("{:.2f} seconds training Q-learning".format(time.time() - t_start))
    # t_start = time.time()
    rewards[1][config] = np.array(causal_q_learning.train())
    del causal_q_learning
    # print("{:.2f} seconds training Q-learning fully informed".format(time.time() - t_start))
    # t_start = time.time()
    rewards[2][config] = np.array(partial_causal_q_learning.train())
    del partial_causal_q_learning
    # print("{:.2f} seconds training Q-learning partially informed".format(time.time() - t_start))
    # t_start = time.time()
    rewards[3][config] = np.array(wrong_causal_q_learning.train())
    del wrong_causal_q_learning
    # print("{:.2f} seconds training Q-learning wrong informed".format(time.time() - t_start))


mean_vectors = [_ for _ in range(4)]
std_dev_vectors = [_ for _ in range(4)]
labels = ["Q-learning", "Q-learning full structure", \
            "Q-learning partial structure", "Q-learning wrong structure"]

for i in range(4):
    mean_vectors[i] = np.mean(rewards[i], axis=0)
    std_dev_vectors[i] = np.std(rewards[i], axis=0)


x_axis = mod_episode * (np.arange(len(mean_vectors[0])))
plot_rewards(x_axis, mean_vectors, std_dev_vectors, labels,\
    "Average reward comparison {} {}".format(num, structure), "plots/comparison_dqn_{}_{}_{}_{}_{}".format(number_of_experiments, num, structure, episodes,"sto" if stochastic else "det"))
np.save("./rewards_data/dqn_{}_{}_{}_{}_{}".format(number_of_experiments, num, structure, episodes,"sto" if stochastic else "det"), rewards)
# a = vanilla_q_learning.test()
# b = causal_q_learning.test()
# c = partial_causal_q_learning.test()
# d = wrong_causal_q_learning.test()
