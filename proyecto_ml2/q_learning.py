# -*- coding: utf-8 -*-
import random
import os
import time

import numpy as np
import gym
import matplotlib.pyplot as plt



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
		# state = self.env.decode(state)
		if random.random() > self.epsilon:
			return np.argmax(self.Q[state, :]), None
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
				# Elige la mejor acción
				action, _ = self.epsilon_greedy(state)
				new_state, reward, done, info = self.env.step(action)
				if done:
					self.Q[new_state] = np.ones(self.number_of_actions) * reward * 10
				self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state, :]) - self.Q[state, action])
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
		state = self.env.reset()
		self.env.render()
		done = False

		while not done:
		  time.sleep(.5)
		  action = np.argmax(self.Q[state])
		  new_state, reward, done, info = self.env.step(action)
		  os.system('clear')
		  self.env.render()
		  state = new_state
		self.env.close()

def main():
	total = []
	for i in range(10):
		time_s = time.time()
		q = QLearning()
		avg = q.train()
		total.append(avg)
	mean_reward = np.mean(total, axis=0)
	for i in mean_reward:
		print(i)
	# q.test()

if __name__ == '__main__':
	main()

