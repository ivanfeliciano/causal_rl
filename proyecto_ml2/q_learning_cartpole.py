# -*- coding: utf-8 -*-
import random
import os
import time
import math

import numpy as np
import gym
import matplotlib.pyplot as plt



class QLearning(object):
	"""docstring for QLearning"""
	def __init__(self, episodes=10000, alpha=0.1, gamma=0.9, epsilon=0.5):
		self.episodes = episodes
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.env = gym.make('CartPole-v1')
		self.number_of_actions = self.env.action_space.n
		# self.number_of_states = self.env.observation_space.n
		self.buckets=(1, 1, 6, 12,)
		self.Q = np.zeros(self.buckets + (self.env.action_space.n,))
	def discretize(self, obs):
		upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
		lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
		ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
		new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
		new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
		return tuple(new_obs)

	def epsilon_greedy(self, state, discretized_state):
		# state = self.env.decode(state)
		if np.random.uniform() > self.epsilon:
			return np.argmax(self.Q[discretized_state]), None
		# Elige una acción aleatoria
		return self.env.action_space.sample(), None
	def train(self, plot_name="rewards_vanilla_q_learning.jpg"):
		state = self.env.reset()
		reward_list = []
		avg_reward_list = []
		for episode in range(self.episodes):
			reward_episode = 0
			state = self.env.reset()
			# print("Episode {}".format(episode))
			done = False
			while not done:
				# self.env.render()
				# Elige la mejor acción
				discretized_state = self.discretize(state)
				action, _ = self.epsilon_greedy(state, discretized_state)
				new_state, reward, done, info = self.env.step(action)
				# if reward == 0:
				# 	reward -= 20
				discretized_new_state = self.discretize(new_state)
				self.Q[discretized_state][action] = self.Q[discretized_state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[discretized_new_state]) - self.Q[discretized_state][action])
				state = new_state
				reward_episode += reward
			reward_list.append(reward_episode)
			# if (episode + 1) % 20 == 0:
			ave_reward = np.mean(reward_list)
			# avg_reward_list.append(ave_reward)
			reward_list = []
			if (episode + 1) % 100 == 0:
			# 	# print('{} {}'.format(episode + 1, ave_reward))
			# 	print('{}'.format(ave_reward))
				avg_reward_list.append(ave_reward)
		return avg_reward_list
		# print("Q")
		# print(self.Q)
		# plt.plot(20*(np.arange(len(avg_reward_list)) + 1), avg_reward_list)
		# plt.xlabel('Episodes')
		# plt.ylabel('Average Reward')
		# plt.title('Average Reward vs Episodes')
		# plt.savefig(plot_name)     
		# plt.close()  
	def test(self):

		while True:
			state = self.env.reset()
			self.env.render()
			done = False

			while not done:
				time.sleep(.1)
				state = self.discretize(state)
				action = np.argmax(self.Q[state])
				new_state, reward, done, info = self.env.step(action)
				# os.system('clear')
				self.env.render()
				state = new_state
		self.env.close()

def main():
	total = []
	# for i in range(10):
	# 	time_s = time.time()
	q = QLearning()
	avg = q.train()
	total.append(avg)
	mean_reward = np.mean(total, axis=0)
	for i in mean_reward:
		print(i)
	q.test()

if __name__ == '__main__':
	main()

