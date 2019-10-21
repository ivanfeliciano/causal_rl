# -*- coding: utf-8 -*-
import random
import os
import time

import numpy as np
import gym
import matplotlib.pyplot as plt
from q_learning import QLearning, MOD_EPISODE
from q_learning_causal import QLearningCausal

def main():
	time_s = time.time()
	episodes = 1000
	number_of_experiments = 50
	total_rewards = [[] for i in range(2)]
	time_to_reach = [[] for i in range(2)]
	for i in range(number_of_experiments):
		print("QLearning")
		rewards_q_learning, episode_t_q = QLearning(episodes=episodes).train(plot_name="QLearning")
		print("QLearningCausal")
		rewards_q_learning_causal, episode_t_q_causal = QLearningCausal(episodes=episodes).train(plot_name="QLearningCausal")
		total_rewards[0].append(rewards_q_learning)
		total_rewards[1].append(rewards_q_learning_causal)
		time_to_reach[0].append(episode_t_q)
		time_to_reach[1].append(episode_t_q_causal)
		# total_rewards[2].append(QLearning(episodes=episodes).train(use_reward_feedback=True))

	scale_x = len(np.mean(total_rewards[0], axis=0))
	plot_x_axis = MOD_EPISODE * (np.arange(scale_x) + 1)
	plt.plot(plot_x_axis, np.mean(total_rewards[0], axis=0), label="Vanilla Q-learning")
	plt.plot(plot_x_axis, np.mean(total_rewards[1], axis=0), label="Q-learning + CM")
	# plt.plot(plot_x_axis, np.mean(total_rewards[2], axis=0), color="red", label="Q-learning + CM + Feedback Revisited")
	plt.xlabel('Episodes')
	plt.ylabel('Average Reward')
	plt.legend()
	plt.title('Average Reward Comparison')
	plt.savefig("comparison.jpg")     
	plt.close()
	for i in time_to_reach[0]:
		print(i)
	for i in time_to_reach[1]:
		print(i)

if __name__ == '__main__':
	main()

