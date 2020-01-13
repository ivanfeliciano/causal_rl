import os
import random
import time
import collections

import numpy as np
import gym
from matplotlib import pyplot as plt

from simple_dqn import DQNAgent
from causal_dqn import CausalDQN


GOAL_STEPS = 500
def run(agent):
	try:
		rewards_dqn = []
		for e in range(episodes):
			state = env.reset()
			state = np.reshape(state, [1, state_size])
			done = False
			step = 0
			while not done:
				step += 1
				action = agent.act(state)
				next_state, reward, done, _ = env.step(action)
				next_state = np.reshape(next_state, [1, state_size])
				agent.remember(state, action, reward, next_state, done)
				state = next_state
			rewards_dqn.append(step)
			if e > 0 and (e + 1) % 100 == 0:
				print("Episode: {} / {}, Score:  {}".format(e + 1, episodes, step))
			agent.replay(32)
	finally:
		agent.save_model()
	return rewards_dqn
def plot_rewards_comparison(reward_dqn, reward_causal_dqn):
	plot_x_axis = np.arange(len(reward_dqn)) + 1
	plt.plot(plot_x_axis, rewards_dqn, label="DQN")
	plt.plot(plot_x_axis, reward_causal_dqn, label="DQN + CM")
	plt.xlabel('Episodes')
	plt.ylabel('# Steps')
	plt.legend()
	plt.savefig("comparisonDQN_CM{}episodes.jpg".format(episodes))
	plt.close()
if __name__ == "__main__":
	env = gym.make('{}-v{}'.format("CartPole", 1))
	episodes = 1000
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = DQNAgent(state_size, action_size, weights_file='cartpole{}.h5'.format(episodes))
	causal_agent = CausalDQN(state_size, action_size, weights_file='causal_cartpole{}.h5'.format(episodes))
	rewards_dqn = run(agent)
	reward_causal_dqn = run(causal_agent)
	dqn_goal_counter = sum([1 for i in rewards_dqn if i >= GOAL_STEPS])
	causal_dqn_goal_counter = sum([1 for i in reward_causal_dqn if i >= GOAL_STEPS])
	print("# of {} steps reached\nDQN = {}\nDQN + CM = {}".format(GOAL_STEPS, dqn_goal_counter, causal_dqn_goal_counter))
	plot_rewards_comparison(rewards_dqn, reward_causal_dqn)
	env.close()