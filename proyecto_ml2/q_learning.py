# -*- coding: utf-8 -*-
import random
import os
import time

import numpy as np
import gym
import matplotlib.pyplot as plt

MOD_EPISODE = 20

class QLearning(object):
	"""docstring for QLearning"""
	def __init__(self, episodes=1500, alpha=0.8, gamma=0.95, epsilon=0.1):
		self.episodes = episodes
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.env = gym.make('Taxi-v2')
		self.number_of_actions = self.env.action_space.n
		self.number_of_states = self.env.observation_space.n
		self.Q = np.zeros([self.number_of_states, self.number_of_actions])

	def epsilon_greedy(self, state):
		eps = np.random.uniform()
		if eps > self.epsilon:
			return np.argmax(self.Q[state, :]), None
		return self.env.action_space.sample(), None
	def train(self, plot_name="QLearning", use_reward_feedback=False):
		state = self.env.reset()
		reward_list = []
		avg_reward_list = []
		for episode in range(self.episodes):
			reward_episode = 0
			state = self.env.reset()
			done = False
			while not done:
				action, _ = self.epsilon_greedy(state)
				new_state, reward, done, info = self.env.step(action)
				if use_reward_feedback and _ != None: reward *= _
				if done:
					self.Q[new_state] = np.ones(self.number_of_actions) * (reward * 10)
				self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state, :]) - self.Q[state, action])
				state = new_state
				reward_episode += reward
			reward_list.append(reward_episode)
			ave_reward = np.mean(reward_list)
			reward_list = []
			if episode == 0 or (episode + 1) % MOD_EPISODE == 0:
				avg_reward_list.append(ave_reward)
		plot_x_axis = MOD_EPISODE * (np.arange(len(avg_reward_list)) + 1)
		plt.plot(plot_x_axis, avg_reward_list, label=plot_name)
		plt.xlabel('Episodes')
		plt.ylabel('Average Reward')
		plt.legend()
		plt.title('Average Reward Comparison')
		plt.savefig(plot_name + ".jpg")     
		plt.close()
		return avg_reward_list
	def test(self, number_of_tests=1):
		for i in range(number_of_tests):
			state = self.env.reset()
			done = False
			while not done:
				os.system('clear')
				self.env.render()
				action = np.argmax(self.Q[state])
				new_state, reward, done, info = self.env.step(action)
				state = new_state
				time.sleep(.5)
		self.env.close()

def main():
	episodes = 1000
	q = QLearning(episodes=episodes)
	average_reward = q.train()
	q.test()

if __name__ == '__main__':
	main()

