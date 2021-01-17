# -*- coding: utf-8 -*-
import random
import numpy as np
import time

import gym

from q_learning_Taxi import QLearning


def is_a_valid_movement(row, col, action):
	"""	[S, N, E, W] """
	dict_directions = {
		(0, 0) : [1, 0, 1, 0], (0, 1) : [1, 0, 0, 1], (0, 2) : [1, 0, 1, 0], (0, 3) : [1, 0, 1, 1], (0, 4) : [1, 0, 0, 1], (1, 0) : [1, 1, 1, 0], (1, 1) : [1, 1, 1, 1], (1, 2) : [1, 1, 1, 1], (1, 3) : [1, 1, 1, 1], (1, 4) : [1, 1, 0, 1], (2, 0) : [1, 1, 1, 0], (2, 1) : [1, 1, 1, 1], (2, 2) : [1, 1, 1, 1], (2, 3) : [1, 1, 1, 1], (2, 4) : [1, 1, 0, 1], (3, 0) : [1, 1, 0, 0], (3, 1) : [1, 1, 1, 0], (3, 2) : [1, 1, 0, 1], (3, 3) : [1, 1, 1, 0], (3, 4) : [1, 1, 0, 1], (4, 0) : [0, 1, 0, 0], (4, 1) : [0, 1, 1, 0], (4, 2) : [0, 1, 0, 1], (4, 3) : [0, 1, 1, 0], (4, 4) : [0, 1, 0, 1]
	}
	return dict_directions[(row, col)][action]

def counterfactual(state, action, outcome):
	destination_locs = [(0,0), (0,4), (4,0), (4,3)]
	row, col, passenger, destination = state
	variables = {"goal" : False, "onGoalDestination" : False, "dropoff" : False, "onPassengerDestination" : False, "pickup" : False, "inTheCab" : False,}
	
	# Pick and place
	if action == 4 and passenger < 4:
		variables["pickup"] = True
	if action == 5:
		variables["dropoff"] = True
	
	# Other needed variables

	if (row, col) == destination_locs[destination]: variables["onGoalDestination"] = True
	if (passenger < 4) and (row, col) == destination_locs[passenger]: variables["onPassengerDestination"] = True

	# Goal and cab
	variables["inTheCab"] = variables["pickup"] and variables["onPassengerDestination"]
	variables["goal"] = (passenger == 4) and variables["dropoff"] and variables["onGoalDestination"]
	return variables[outcome]


class QLearningCausal(QLearning):
	"""docstring for QLearningCausal"""
	def epsilon_greedy(self, state):
		eps = np.random.uniform()
		# print(state)
		state_decoded = [i for i in self.env.decode(state)]
		# print(state_decoded)
		if eps > self.get_current_value():
			return np.argmax(self.Q[state, :])
		if counterfactual(state_decoded, 5, "goal"):
			# print("Debo dejarlo")
			return 5
		if counterfactual(state_decoded, 4, "inTheCab"):
			# print("Debo subirlo")
			return 4
		return np.random.choice(4)
def main():
	# q = QLearningCausal()
	# avg = q.train("Q-Learning con grafo causal")
	# q.test()
	env = gym.make('Taxi-v3')
	episodes=1
	mod = 1
	nb_steps = 200 * episodes
	q = QLearningCausal(env, episodes=episodes, mod=mod, nb_steps=nb_steps)
	rewards = q.train()
	print(rewards)
	q = QLearning(env, episodes=episodes, mod=mod, nb_steps=nb_steps)
	rewards = q.train()
	print(rewards)

if __name__ == '__main__':
	main()