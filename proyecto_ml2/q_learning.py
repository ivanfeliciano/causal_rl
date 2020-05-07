# -*- coding: utf-8 -*-
import random
import os
import time

import numpy as np
import gym
import matplotlib.pyplot as plt

THRESHOLD = -75
MOD_EPISODE = 1

class QLearning(object):
	"""docstring for QLearning"""
	def __init__(self, episodes=1500, alpha=0.8, gamma=0.95, epsilon=1.0):
		self.episodes = episodes
		self.alpha = alpha
		self.gamma = gamma
		self.true_action_prob = 0.7
		self.epsilon = epsilon
		self.env = gym.make('Taxi-v3')
		self.number_of_actions = self.env.action_space.n
		self.number_of_states = self.env.observation_space.n
		self.Q = np.zeros([self.number_of_states, self.number_of_actions])
		self.eps_min = 0.01
		self.eps_decay = 0.995

	def epsilon_greedy(self, state):
		eps = np.random.uniform()
		if eps > self.epsilon:
			return np.argmax(self.Q[state, :]), None
		return self.env.action_space.sample(), None
	def train(self, plot_name="QLearning", use_reward_feedback=False, stochastic=False):
		state = self.env.reset()
		optimal_reward_episodes = []
		list_of_individual_episode_reward = []
		avg_reward_all_training = []
		flag = True
		threshold_reached = False
		time_to_reach_t = None
		for episode in range(self.episodes):
			reward_episode = 0
			state = self.env.reset()
			done = False
			while not done:
				action, _ = self.epsilon_greedy(state)
				if stochastic:
					if np.random.uniform() > self.true_action_prob:
						remain_actions = [i for i in range(self.number_of_actions)]
						remain_actions.remove(action)
						action = np.random.choice(remain_actions)
				new_state, reward, done, info = self.env.step(action)
				if new_state == state and action < 4:
					reward = -10
				if use_reward_feedback and _ != None: reward *= _
				if done: self.Q[new_state] = np.ones(self.number_of_actions) * (reward * 10)
				self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state, :]) - self.Q[state, action])
				if self.epsilon > self.eps_min:
					self.epsilon *= self.eps_decay
				state = new_state
				reward_episode += reward
			# if not threshold_reached and reward_episode >= THRESHOLD:
			# 	threshold_reached = True
			# 	time_to_reach_t = episode
			if reward_episode >= THRESHOLD:
				optimal_reward_episodes.append(episode)
			list_of_individual_episode_reward.append(reward_episode)
			if episode == 0 or (episode + 1) % MOD_EPISODE == 0:
				ave_reward = np.mean(list_of_individual_episode_reward)
				avg_reward_all_training.append(ave_reward)
				list_of_individual_episode_reward = []
		# print(optimal_reward_episodes)
		plot_x_axis = MOD_EPISODE * (np.arange(len(avg_reward_all_training)) + 1)
		plt.plot(plot_x_axis, avg_reward_all_training, label=plot_name)
		plt.xlabel('Episodes')
		plt.ylabel('Average Reward')
		plt.legend()
		plt.title('Average Reward Comparison')
		plt.savefig(plot_name + ".jpg")     
		plt.close()
		# print("Time to reach t =  {}".format(time_to_reach_t))
		return avg_reward_all_training, optimal_reward_episodes
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

