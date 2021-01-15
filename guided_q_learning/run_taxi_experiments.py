# -*- coding: utf-8 -*-
import argparse
import copy

from scipy import stats
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gym
from statsmodels.stats.power import TTestIndPower

from q_learning_Taxi import QLearning
from q_learning_causal_Taxi import QLearningCausal
from utils.vis_utils import plot_rewards
from utils.performance_utils import cohend, welch_ttest



def test(x, y):
	# stat, p = stats.mannwhitneyu(x, y)
	stat, p = stats.ttest_ind(x, y, equal_var=False)
	print('Statistics=%.3f, p=%.3f' % (stat, p))
	# interpret
	alpha = 0.05
	if p > alpha:
		print('Same distribution (fail to reject H0)')
	else:
		print('Different distribution (reject H0)')
	effect = cohend(x, y)
	alpha = 0.05
	analysis = TTestIndPower()
	power = analysis.solve_power(effect, power=None, nobs1=len(x), ratio=1.0, alpha=alpha)
	samples = analysis.solve_power(effect, power=0.8, nobs1=None, ratio=1.0, alpha=alpha)
	dof = welch_ttest(np.array(x), np.array(y))
	print(f"effect: {effect:.2f}")
	print(f"power: {power:.2f}")
	print(f"samples: {samples:.2f}")
	print(f"degrees of freedom: {dof:.2f}")

def longest_streak(list_episodes):
	streak_start = 0
	length = 0
	max_length = 0
	best_start = 0
	expected_len = 50
	for i in range(len(list_episodes) - 1):
		if list_episodes[i] == list_episodes[i + 1] - 1:
			length += 1
			max_length = max(max_length, length)
		else:
			if max_length < length and not max_length >= 100:
				best_start = streak_start
				max_length = length
			length = 0
			streak_start = i
	if max_length < expected_len:
		return 1000, max_length
	return streak_start, max_length



def main():
	parser = argparse.ArgumentParser(description='Run Q-learning and Q-learning CM to solve the classic Taxi RL problem.')
	parser.add_argument("--stochastic", help="change to simple stochastic enviroments (0.7 prob of do the choosen action)",\
						action="store_true")
	parser.add_argument("--episodes", type=int, default=1000, help="# of episodes per experiment")
	parser.add_argument("--partition", type=int, default=100, help="eps factor")
	parser.add_argument("--experiments", type=int, default=100, help="# of experiments")
	parser.add_argument("--mod", type=int, default=50, help="# module episodes")
	parser.add_argument("--threshold", type=int, default=0, help="# module episodes")
	parser.add_argument("-plt", "--plot", action="store_true", help="Plot reward comparison")
	parser.add_argument("-v", "--verbose", action="store_true", help="Verbosite activated")
	args = parser.parse_args()
	episodes = args.episodes
	mod = args.mod
	threshold = args.threshold
	print(threshold)
	stochastic = args.stochastic
	number_of_experiments = args.experiments
	total_rewards = [[] for i in range(2)]
	partition = args.partition
	np.random.seed(0)
	nb_steps = 200 * episodes * partition // 100
	pbar = tqdm(range(number_of_experiments))
	pbar.set_description("Processing experiment")
	optimal_reward_streak_vanilla_init = []
	optimal_reward_streak_vanilla_size = []
	optimal_reward_streak_causal_init = []
	optimal_reward_streak_causal_size = []
	mean_opt = 0
	mean_opt_c = 0
	for i in pbar:
		env = gym.make('Taxi-v3')
		env_ = copy.deepcopy(env)
		q_agent = QLearning(env, episodes=episodes, mod=mod, nb_steps=nb_steps)
		q_agent_causal = QLearningCausal(env_, episodes=episodes, mod=mod, nb_steps=nb_steps)
		if stochastic:
			q_agent.threshold = threshold
			q_agent_causal.threshold = threshold
		rewards_q_learning = q_agent.train(stochastic=stochastic)
		rewards_q_learning_causal = q_agent_causal.train(stochastic=stochastic)
		total_rewards[0].append(rewards_q_learning)
		total_rewards[1].append(rewards_q_learning_causal)
		vanilla_init, vanilla_size = longest_streak(q_agent.optimal_reward_episodes)
		causal_init, causal_size = longest_streak(q_agent_causal.optimal_reward_episodes)
		optimal_reward_streak_vanilla_init.append(vanilla_init)
		optimal_reward_streak_vanilla_size.append(vanilla_size)
		optimal_reward_streak_causal_init.append(causal_init)
		optimal_reward_streak_causal_size.append(causal_size)
	len_vectors = 2
	mean_vectors = [_ for _ in range(len_vectors)]
	std_dev_vectors = [_ for _ in range(len_vectors)]
	labels = ["Q-learning", "Q-learning con estructura causal"]#, "Recompensa óptima"]
	for i in range(2):
		mean_vectors[i] = np.mean(total_rewards[i], axis=0)
		std_dev_vectors[i] = np.std(total_rewards[i], axis=0)
	# std_dev_vectors[2] = np.zeros(len(mean_vectors[0]))
	# mean_vectors[2] = np.ones(len(mean_vectors[0]))
	# if stochastic:
	# 	mean_vectors[2] = mean_vectors[2] * 0
	# else:
	# 	mean_vectors[2] = mean_vectors[2] * 9.7
	print("Mean causal initial episode {}".format(np.mean(optimal_reward_streak_causal_init)))
	print("Std causal initial episode {}".format(np.std(optimal_reward_streak_causal_init)))
	print("Mean vanilla initial episode {}".format(np.mean(optimal_reward_streak_vanilla_init)))
	print("std vanilla initial episode {}".format(np.std(optimal_reward_streak_vanilla_init)))
	print(optimal_reward_streak_causal_init)
	print(optimal_reward_streak_causal_size)
	print(optimal_reward_streak_vanilla_init)
	print(optimal_reward_streak_vanilla_size)
	test(optimal_reward_streak_vanilla_init, optimal_reward_streak_causal_init)
	

	# print("Mean causal streak size {}".format(np.mean(optimal_reward_streak_causal_size)))
	# print("std causal streak size {}".format(np.std(optimal_reward_streak_causal_size)))
	# print("Mean vanilla streak size {}".format(np.mean(optimal_reward_streak_vanilla_size)))
	# print("std vanilla streak size {}".format(np.std(optimal_reward_streak_vanilla_size)))
	# test(optimal_reward_streak_vanilla_size, optimal_reward_streak_causal_size)
	x_axis = mod * (np.arange(len(mean_vectors[1])))
	plot_dir_name = "plots/taxi/"
	plot_name = "taxi_env_comparison_{}_{}_{}_eps_{}".format("sto" if stochastic else "det", episodes, number_of_experiments, nb_steps)
	if args.plot:
		plot_rewards(x_axis, mean_vectors, std_dev_vectors, labels, plot_dir_name + plot_name)
if __name__ == '__main__':
	main()

