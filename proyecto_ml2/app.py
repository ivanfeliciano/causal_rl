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
	number_of_experiments = 10
	total_rewards = [[] for i in range(3)]
	for i in range(number_of_experiments):
		total_rewards[0].append(QLearning(episodes=episodes).train())
		total_rewards[1].append(QLearningCausal(episodes=episodes).train())
		total_rewards[2].append(QLearning(episodes=episodes).train(use_reward_feedback=True))
	scale_x = len(np.mean(total_rewards[0], axis=0))
	plot_x_axis = MOD_EPISODE * (np.arange(scale_x) + 1)
	plt.plot(plot_x_axis, np.mean(total_rewards[0], axis=0), label="Vanilla Q-learning")
	plt.plot(plot_x_axis, np.mean(total_rewards[1], axis=0), label="Q-learning + CM")
	plt.plot(plot_x_axis, np.mean(total_rewards[2], axis=0), color="red", label="Q-learning + CM + Feedback")
	plt.xlabel('Episodes')
	plt.ylabel('Average Reward')
	plt.legend()
	plt.title('Average Reward Comparison')
	plt.savefig("comparison.jpg")     
	plt.close()

if __name__ == '__main__':
	main()

