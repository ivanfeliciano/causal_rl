# -*- coding: utf-8 -*-
import argparse
import copy

from scipy import stats
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gym

from q_learning_Taxi import QLearning
from q_learning_causal_Taxi import QLearningCausal
from utils.vis_utils import plot_rewards

def main():
	parser = argparse.ArgumentParser(description='Run Q-learning and Q-learning CM to solve the classic Taxi RL problem.')
	parser.add_argument("--stochastic", help="change to simple stochastic enviroments (0.7 prob of do the choosen action)",\
						action="store_true")
	parser.add_argument("--episodes", type=int, default=1000, help="# of episodes per experiment")
	parser.add_argument("--experiments", type=int, default=100, help="# of experiments")
	parser.add_argument("--mod", type=int, default=50, help="# module episodes")
	parser.add_argument("-plt", "--plot", action="store_true", help="Plot reward comparison")
	parser.add_argument("-v", "--verbose", action="store_true", help="Verbosite activated")
	args = parser.parse_args()
	episodes = args.episodes
	mod = args.mod
	stochastic = args.stochastic
	number_of_experiments = args.experiments
	total_rewards = [[] for i in range(2)]
	np.random.seed(0)
	nb_steps = 200 * episodes // 50
	pbar = tqdm(range(number_of_experiments))
	pbar.set_description("Processing experiment")
	for i in pbar:
		env = gym.make('Taxi-v3')
		env_ = copy.deepcopy(env)
		rewards_q_learning = QLearning(env, episodes=episodes, mod=mod, nb_steps=nb_steps).train(stochastic=stochastic)
		rewards_q_learning_causal = QLearningCausal(env_, episodes=episodes, mod=mod, nb_steps=nb_steps).train(stochastic=stochastic)
		total_rewards[0].append(rewards_q_learning)
		total_rewards[1].append(rewards_q_learning_causal)
	mean_vectors = [_ for _ in range(3)]
	std_dev_vectors = [_ for _ in range(3)]
	labels = ["Q-learning", "Q-learning full structure", "Optimal reward"]
	for i in range(2):
		mean_vectors[i] = np.mean(total_rewards[i], axis=0)
		std_dev_vectors[i] = np.std(total_rewards[i], axis=0)
	mean_vectors[2] = np.ones(len(mean_vectors[0])) * 8
	std_dev_vectors[2] = np.zeros(len(mean_vectors[0]))

	x_axis = mod * (np.arange(len(mean_vectors[1])))
	plot_dir_name = "plots/taxi/"
	plot_name = "taxi_env_comparison_{}_{}_{}_eps_{}".format("sto" if stochastic else "det", episodes, number_of_experiments, nb_steps)
	if args.plot:
		plot_rewards(x_axis, mean_vectors, std_dev_vectors, labels,\
						"Average reward", plot_dir_name + plot_name)	
if __name__ == '__main__':
	main()

