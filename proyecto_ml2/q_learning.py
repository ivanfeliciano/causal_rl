# -*- coding: utf-8 -*-
import random
import os
import time

import numpy as np
import gym
import matplotlib.pyplot as plt

THRESHOLD = 9
MOD_EPISODE = 1
random.seed(42)

class QLearning(object):
	"""docstring for QLearning"""
	def __init__(self, episodes=1500, alpha=0.8, gamma=0.95, epsilon=0.2):
		self.episodes = episodes
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.env = gym.make('Taxi-v2')
		self.number_of_actions = self.env.action_space.n
		self.number_of_states = self.env.observation_space.n
		self.random_actions = 0
		self.Q = np.zeros([self.number_of_states, self.number_of_actions])

	def epsilon_greedy(self, state):
		eps = np.random.uniform()
		if eps > self.epsilon:
			return np.argmax(self.Q[state, :]), None
		self.random_actions += 1
		return self.env.action_space.sample(), None
	def train(self, plot_name="QLearning", use_reward_feedback=False, stochastic=False):
		state = self.env.reset()
		reward_list = []
		avg_reward_list = []
		times_stuckit = 0
		flag = True
		threshold_reached = False
		time_to_reach_t = None
		for episode in range(self.episodes):
			reward_episode = 0
			state = self.env.reset()
			done = False
			while not done:
				action, _ = self.epsilon_greedy(state)
				new_state, reward, done, info = self.env.step(action)
				if new_state == state and action < 4:
					# if flag:
					# 	print("new_state {}, state {}".format([i for i in self.env.decode(new_state)], [i for i in self.env.decode(state)]))
					# 	print("action {}".format(action))
					# 	flag = False
					times_stuckit += 1
					reward = -10
				if use_reward_feedback and _ != None: reward *= _
				if done: self.Q[new_state] = np.ones(self.number_of_actions) * (reward * 10)
				self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state, :]) - self.Q[state, action])
				state = new_state
				reward_episode += reward
			if not threshold_reached and reward_episode >= THRESHOLD:
				threshold_reached = True
				time_to_reach_t = episode
			reward_list.append(reward_episode)
			if episode == 0 or (episode + 1) % MOD_EPISODE == 0:
				# print(reward_list)
				ave_reward = np.mean(reward_list)
				avg_reward_list.append(ave_reward)
				reward_list = []
		plot_x_axis = MOD_EPISODE * (np.arange(len(avg_reward_list)) + 1)
		plt.plot(plot_x_axis, avg_reward_list, label=plot_name)
		plt.xlabel('Episodes')
		plt.ylabel('Average Reward')
		plt.legend()
		plt.title('Average Reward Comparison')
		plt.savefig(plot_name + ".jpg")     
		plt.close()
		print(times_stuckit)
		print(self.random_actions)
		print("Time to reach t =  {}".format(time_to_reach_t))
		return avg_reward_list, time_to_reach_t
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

