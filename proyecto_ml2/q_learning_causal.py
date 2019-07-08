# -*- coding: utf-8 -*-
import random

import numpy as np
from structural_equation_modeling import TaxiSEM
from q_learning import QLearning


def is_a_valid_movement(row, col, action):
	"""	[S, N, E, W] """
	dict_directions = {
		(0, 0) : [1, 0, 1, 0], (0, 1) : [1, 0, 0, 1], (0, 2) : [1, 0, 1, 0], (0, 3) : [1, 0, 1, 1], (0, 4) : [1, 0, 0, 1], (1, 0) : [1, 1, 1, 0], (1, 1) : [1, 1, 1, 1], (1, 2) : [1, 1, 1, 1], (1, 3) : [1, 1, 1, 1], (1, 4) : [1, 1, 0, 1], (2, 0) : [1, 1, 1, 0], (2, 1) : [1, 1, 1, 1], (2, 2) : [1, 1, 1, 1], (2, 3) : [1, 1, 1, 1], (2, 4) : [1, 1, 0, 1], (3, 0) : [1, 1, 0, 0], (3, 1) : [1, 1, 1, 0], (3, 2) : [1, 1, 0, 1], (3, 3) : [1, 1, 1, 0], (3, 4) : [1, 1, 0, 1], (4, 0) : [0, 1, 0, 0], (4, 1) : [0, 1, 1, 0], (4, 2) : [0, 1, 0, 1], (4, 3) : [0, 1, 1, 0], (4, 4) : [0, 1, 0, 1]
	}
	return dict_directions[(row, col)][action]

def counterfactual(state, action, outcome):
	destination_locs = [(0,0), (0,4), (4,0), (4,3)]
	row, col, passenger, destination = state
	variables = {
		"southA" : False, "northA" : False, "westA" : False, "eastA" : False, "southMove" : False, "northMove" : False, "eastMove" : False, "westMove" : False, "goal" : False, "onGoalDestination" : False, "dropoff" : False, "onPassengerDestination" : False, "pickup" : False, "inTheCab" : False,}
	# Movements
	if action == 0 and is_a_valid_movement(row, col, action):
		variables["southMove"] = True
	if action == 1 and  is_a_valid_movement(row, col, action):
		variables["northMove"] = True
	if action == 2 and is_a_valid_movement(row, col, action):
		variables["eastMove"]
	if action == 3 and is_a_valid_movement(row, col, action):
		variables["westMove"]
	
	# Pick and place
	if action == 4:
		variables["pickup"] = True
	if action == 5:
		variables["dropoff"] = True
	
	# Other needed variables

	if (row, col) == destination_locs[destination]: variables["onGoalDestination"] = True
	if (passenger < 4) and (row, col) == destination_locs[passenger]: variables["onPassengerDestination"] = True

	# Goal and cab
	variables["inTheCab"] = (passenger > 3) or ( variables["pickup"] and variables["onPassengerDestination"])
	variables["goal"] = variables["inTheCab"] and variables["dropoff"] and variables["onGoalDestination"]
	return variables[outcome]

class QLearningCausal(QLearning):
	"""docstring for QLearningCausal"""
	def epsilon_greedy(self, state):
		# ignore_oracle = True if random.random() > 0.99 else False
		ignore_oracle = False
		if random.random() < 0.9:
			return np.argmax(self.Q[state, :]), None
		state_decoded = [i for i in self.env.decode(state)]
		if counterfactual(state_decoded, 5, "goal"):
			return 5, 10
		if counterfactual(state_decoded, 4, "inTheCab"):
			return 4, 5
		possible_moves = []
		if counterfactual(state_decoded, 0, "southMove"): possible_moves.append(0)
		if counterfactual(state_decoded, 1, "northMove"): possible_moves.append(1)
		if counterfactual(state_decoded, 2, "eastMove"): possible_moves.append(2)
		if counterfactual(state_decoded, 3, "westMove"): possible_moves.append(3)
		best = [-1000, -1000, -1000, -1000]
		for i in possible_moves:
			best[i] = self.Q[state][i]
		return np.argmax(best), None
		# return self.env.action_space.sample()
def main():
	import time
	total = []
	# for i in range(10):
	# 	time_s = time.time()
	q = QLearningCausal()
	avg = q.train()
		# q.train()
	q.test()
	total.append(avg)
	mean_reward = np.mean(total, axis=0)
	for i in mean_reward:
		print(i)

if __name__ == '__main__':
	main()