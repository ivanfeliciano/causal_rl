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
parser.add_argument("--stochastic", help="change to simple stochastic enviroments (0.75 prob of do the choosen action)",\
                    action="store_true")
parser.add_argument("--episodes", type=int, default=100, help="# of episodes per experiment")
parser.add_argument("--basedir", type=str, default="/home/ivan/Documentos/causal_rl/guided_q_learning/results_thesis/light_switches", help="path to save results")
parser.add_argument("--mod", type=int, default=20, help="# block of avg episodes")
parser.add_argument("--experiments", type=int, default=1, help="# of experiments, one experiment corresponds to one structure and one goal")
parser.add_argument("--num", type=int, default=5, help="Number of switches")
parser.add_argument("--structure", type=str, default="one_to_one", help="structure of the graph")
parser.add_argument("-d", "--draw", action="store_true", help="Draw graphs activated")
parser.add_argument("--pmod", type=str, default="low", help="percentage of incorrectness")
parser.add_argument("--partition", type=int, default=100, help="at what percentage begins with small epsilon")
args = parser.parse_args()

base_dir = args.basedir
mod_episode = args.mod
partition = args.partition
num = args.num 
episodes = args.episodes
structure = args.structure
draw = args.draw
number_of_experiments = args.experiments
stochastic = args.stochastic
pmod = args.pmod
model = load_model('models/multilabel_classifier84.h5')

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
nb_steps = num * episodes * partition // 100


env = LightEnv(horizon=horizon, num=num, structure=structure)
rewards = np.array([[None for _ in range(number_of_experiments)] for _ in range(n_algorithms)])
processor = LightEnvProcessor()
file_name = "{}_{}_{}_{}_N_{}_experiments_{}_episodes_{}_eps_{}".format(env_config, pmod, pmod_factor, structure, num, number_of_experiments, episodes, partition)

for config in range(number_of_experiments):
    env.keep_struct = False
    env.reset()
    env.keep_struct = True
    print("Running training on new struct and new goal. {} / {}".format(config + 1, number_of_experiments))

    full_adj_list = aj_to_adj_list(env.aj)
    incomplete_adj_list = del_edges(deepcopy(full_adj_list), pmod_factor)
    incorrect_adj_list = shuffle_aj_mat(deepcopy(full_adj_list), pmod_factor)
    print("finished adj mat init")

    vanilla_env = LightAndSwitchEnv(copy(env), full_adj_list, discrete=False)
    full_info_environment = LightAndSwitchEnv(copy(env), full_adj_list, discrete=False)
    partial_info_environment = LightAndSwitchEnv(copy(env), incomplete_adj_list, discrete=False)
    wrong_info_environment = LightAndSwitchEnv(copy(env), incorrect_adj_list, discrete=False)

    if draw:
        full_info_environment.causal_structure.draw_graph("{}/graphs/full_{}_{}".format(base_dir, config, file_name)) 
        partial_info_environment.causal_structure.draw_graph("{}/graphs/partial_{}_{}".format(base_dir, config, file_name)) 
        wrong_info_environment.causal_structure.draw_graph("{}/graphs/wrong_{}_{}".format(base_dir, config, file_name))


    eps_policy = LinearAnnealedPolicy(EpsilonGreedyDQN(vanilla_env, num, None, False),\
                                            attr='eps', value_max=1., value_min=.1,\
                                            value_test=.05, nb_steps=nb_steps)
    vanilla_q_learning = DQN(vanilla_env, num + 1, processor, eps_policy, episodes, mod_episode=mod_episode)
    t_start = time.time()
    print("Running vanilla_q_learning")
    rewards[0][config] = np.array(vanilla_q_learning.train())
    del vanilla_q_learning
    del eps_policy
    print("{:.2f} seconds training Q-learning".format(time.time() - t_start))
    
    
    causal_eps_policy = LinearAnnealedPolicy(EpsilonGreedyDQN(full_info_environment, num, model, True),\
                                            attr='eps', value_max=1., value_min=.1,\
                                            value_test=.05, nb_steps=nb_steps)
    causal_q_learning = DQN(full_info_environment, num + 1, processor, causal_eps_policy, episodes, mod_episode=mod_episode)
    t_start = time.time()
    print("Running causal_q_learning")
    rewards[1][config] = np.array(causal_q_learning.train())
    del causal_q_learning
    del causal_eps_policy
    print("{:.2f} seconds training Q-learning fully informed".format(time.time() - t_start))


    partial_causal_eps_policy = LinearAnnealedPolicy(EpsilonGreedyDQN(partial_info_environment, num, model, True),\
                                            attr='eps', value_max=1., value_min=.1,\
                                            value_test=.05, nb_steps=nb_steps)
    partial_causal_q_learning = DQN(partial_info_environment, num + 1, processor, partial_causal_eps_policy, episodes, mod_episode=mod_episode)
    t_start = time.time()
    print("Running partial_causal_q_learning")
    rewards[2][config] = np.array(partial_causal_q_learning.train())
    del partial_causal_q_learning
    del partial_causal_eps_policy
    print("{:.2f} seconds training Q-learning partially informed".format(time.time() - t_start))
    

    wrong_causal_eps_policy = LinearAnnealedPolicy(EpsilonGreedyDQN(wrong_info_environment, num, model, True),\
                                            attr='eps', value_max=1., value_min=.1,\
                                            value_test=.05, nb_steps=nb_steps)                                
    wrong_causal_q_learning = DQN(wrong_info_environment, num + 1, processor, wrong_causal_eps_policy, episodes, mod_episode=mod_episode)
    t_start = time.time()
    print("Running wrong_causal_q_learning")
    rewards[3][config] = np.array(wrong_causal_q_learning.train())
    del wrong_causal_q_learning
    print("{:.2f} seconds training Q-learning wrong informed".format(time.time() - t_start))


mean_vectors = [_ for _ in range(n_algorithms)]
std_dev_vectors = [_ for _ in range(n_algorithms)]
labels = ["DQN", "DQN + estructura completa", \
            "DQN + estructura parcial", "DQN + estructura incorrecta"]

for i in range(n_algorithms):
    mean_vectors[i] = np.mean(rewards[i], axis=0)
    std_dev_vectors[i] = np.std(rewards[i], axis=0)

x_axis = mod_episode * (np.arange(len(mean_vectors[0])))

np.save("{}/rewards_mats/{}".format(base_dir, file_name), rewards)
plot_path = "{}/plots/{}".format(base_dir, file_name)

plot_rewards(x_axis, mean_vectors, std_dev_vectors, labels, plot_path)
