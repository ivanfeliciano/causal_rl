# -*- coding: utf-8 -*-
import random
import numpy as np
import time
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
	# variables = {"southA" : False, "northA" : False, "westA" : False, "eastA" : False, "southMove" : False, "northMove" : False, "eastMove" : False, "westMove" : False, "goal" : False, "onGoalDestination" : False, "dropoff" : False, "onPassengerDestination" : False, "pickup" : False, "inTheCab" : False,}
	# # Movements
	# if action == 0 and is_a_valid_movement(row, col, action):
	# 	variables["southMove"] = True
	# if action == 1 and  is_a_valid_movement(row, col, action):
	# 	variables["northMove"] = True
	# if action == 2 and is_a_valid_movement(row, col, action):
	# 	variables["eastMove"] = True
	# if action == 3 and is_a_valid_movement(row, col, action):
	# 	variables["westMove"] = True
	
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
		# if self.is_test: self.epsilon = self.eps_min
		if counterfactual(state_decoded, 5, "goal"):
			# print("*****************")
			# print("I should dropoff")
			# print(state_decoded)
			# self.env.render()
			# print("*****************")
			return 5
		if counterfactual(state_decoded, 4, "inTheCab"):
			# print("*****************")
			# print("I should pick up")
			# print(state_decoded)
			# self.env.render()
			# print("*****************")
			return 4
		if eps > self.get_current_value():
			return np.argmax(self.Q[state, :4])
		return np.random.choice(4)
def main():
	q = QLearningCausal()
	avg = q.train("Q-Learning con grafo causal")
	q.test()

if __name__ == '__main__':
	main()