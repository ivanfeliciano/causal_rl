# -*- coding: utf-8 -*-
import random
import os
import time

import numpy as np
import gym
import matplotlib.pyplot as plt

# THRESHOLD = -75
# MOD_EPISODE = 1

class QLearning(object):
	"""docstring for QLearning"""
	def __init__(self, env, episodes=1500, alpha=0.8, gamma=0.95, epsilon=1.0, mod=50, nb_steps=100):
		self.episodes = episodes
		self.alpha = alpha
		self.gamma = gamma
		self.true_action_prob = 0.7
		self.epsilon = epsilon
		self.env = env
		self.number_of_actions = self.env.action_space.n
		self.number_of_states = self.env.observation_space.n
		self.Q = np.zeros([self.number_of_states, self.number_of_actions])
		self.eps_min = 0.1
		self.eps_decay = 0.995
		self.mod = mod
		self.eps_max = 1.0
		self.is_test = False
		self.nb_steps = nb_steps

	def get_current_value(self, training=True):
		if not self.is_test:
			a = -float(self.eps_max - self.eps_min) / float(self.nb_steps)
			b = float(self.eps_max)
			value = max(self.eps_min, a * float(self.step) + b)
		else:
			value = self.eps_min
		# self.step = (self.step +  1) % self.nb_steps
		self.step += 1
		# print("eps = {}".format(value))
		return value

	def epsilon_greedy(self, state):
		eps = np.random.uniform()
		# if self.is_test: self.epsilon = self.eps_min
		if eps > self.get_current_value():
			return np.argmax(self.Q[state, :])
		return np.random.choice(6)
	def train(self, stochastic=False):
		state = self.env.reset()
		optimal_reward_episodes = []
		list_of_individual_episode_reward = []
		avg_reward_all_training = []
		flag = True
		threshold_reached = False
		self.step = 0
		for episode in range(self.episodes):
			# print("EPISODE {}".format(episode))
			reward_episode = 0
			np.random.seed(0)
			state = self.env.reset()
			done = False
			self.epsilon = 1
			while not done:
				action = self.epsilon_greedy(state)
				if stochastic:
					if np.random.uniform() > self.true_action_prob:
						remain_actions = [i for i in range(self.number_of_actions)]
						remain_actions.remove(action)
						action = np.random.choice(remain_actions)
				new_state, reward, done, info = self.env.step(action)
				self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state, :]) - self.Q[state, action])
				self.epsilon = max(self.eps_decay * self.epsilon, self.eps_min)
				state = new_state
				reward_episode += reward
			# if not threshold_reached and reward_episode >= THRESHOLD:
			# 	threshold_reached = True
			# if reward_episode >= THRESHOLD:
			# 	optimal_reward_episodes.append(episode)
			list_of_individual_episode_reward.append(reward_episode)
			if episode == 0 or (episode + 1) % self.mod == 0:
				ave_reward = np.mean(list_of_individual_episode_reward)
				avg_reward_all_training.append(ave_reward)
				list_of_individual_episode_reward = []
		# print(optimal_reward_episodes)
		# plot_x_axis = self. * (np.arange(len(avg_reward_all_training)) + 1)
		# plt.plot(plot_x_axis, avg_reward_all_training, label=plot_name)
		# plt.xlabel('Episodes')
		# plt.ylabel('Average Reward')
		# plt.legend()
		# plt.title('Average Reward Comparison')
		# plt.savefig(plot_name + ".jpg")     
		# plt.close()
		return avg_reward_all_training
	def test(self, number_of_tests=1):
		list_of_individual_episode_reward = []
		avg_reward_all_training = []
		self.is_test = True
		for episode in range(number_of_tests):
			reward_episode = 0
			state = self.env.reset()
			done = False
			while not done:
				action = self.epsilon_greedy(state)
				new_state, reward, done, info = self.env.step(action)
				state = new_state
				reward_episode += reward
			list_of_individual_episode_reward.append(reward_episode)
			if episode == 0 or (episode + 1) % self.mod == 0:
				ave_reward = np.mean(list_of_individual_episode_reward)
				avg_reward_all_training.append(ave_reward)
				list_of_individual_episode_reward = []
		self.env.close()
		return avg_reward_all_training

def main():
	episodes = 1000
	q = QLearning(episodes=episodes)
	average_reward = q.train()
	q.test()

if __name__ == '__main__':
	main()

