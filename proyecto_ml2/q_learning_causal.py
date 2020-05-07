# -*- coding: utf-8 -*-
import random
import numpy as np
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
	variables = {"southA" : False, "northA" : False, "westA" : False, "eastA" : False, "southMove" : False, "northMove" : False, "eastMove" : False, "westMove" : False, "goal" : False, "onGoalDestination" : False, "dropoff" : False, "onPassengerDestination" : False, "pickup" : False, "inTheCab" : False,}
	# Movements
	if action == 0 and is_a_valid_movement(row, col, action):
		variables["southMove"] = True
	if action == 1 and  is_a_valid_movement(row, col, action):
		variables["northMove"] = True
	if action == 2 and is_a_valid_movement(row, col, action):
		variables["eastMove"] = True
	if action == 3 and is_a_valid_movement(row, col, action):
		variables["westMove"] = True
	
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
		state_decoded = [i for i in self.env.decode(state)]
		possible_moves = []
		if counterfactual(state_decoded, 5, "goal"):
			return 5, 2
		if counterfactual(state_decoded, 4, "inTheCab"):
			return 4, 1.5
		if eps > self.epsilon:
			best = [-100000, -100000, -100000, -100000]
			if counterfactual(state_decoded, 0, "southMove"): best[0] = self.Q[state][0]
			if counterfactual(state_decoded, 1, "northMove"): best[1] = self.Q[state][1]
			if counterfactual(state_decoded, 2, "eastMove"): best[2] = self.Q[state][2]
			if counterfactual(state_decoded, 3, "westMove"): best[3] = self.Q[state][3]
			if best != [-100000, -100000, -100000, -100000]:
				return np.argmax(best), -1
			return np.argmax(self.Q[state, :]), None
		best = []
		if counterfactual(state_decoded, 0, "southMove"): best.append(0)
		if counterfactual(state_decoded, 1, "northMove"): best.append(1)
		if counterfactual(state_decoded, 2, "eastMove"): best.append(2)
		if counterfactual(state_decoded, 3, "westMove"): best.append(3)
		if len(best) > 0:
			return random.choice(best), -1
		return self.env.action_space.sample(), None
def main():
	q = QLearningCausal()
	avg = q.train("QLearningCausal")
	q.test()

if __name__ == '__main__':
	main()