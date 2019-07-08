# -*- coding: utf-8 -*-
import random
import math

import numpy as np
from q_learning_cartpole import QLearning

def counterfactual(state, action):
	theta_threshold_radians = 12 * 2 * math.pi / 360
	x_threshold = 2.4
	x, x_dot, theta, theta_dot = state
	gravity = 9.8
	masscart = 1.0
	masspole = 0.1
	total_mass = (masspole + masscart)
	length = 0.5 # actually half the pole's length
	polemass_length = (masspole * length)
	force_mag = 10.0
	tau = 0.02  # seconds between state updates

	force = force_mag if action == 1 else -force_mag
	costheta = math.cos(theta)
	sintheta = math.sin(theta)
	temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
	thetaacc = (gravity * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
	xacc  = temp - polemass_length * thetaacc * costheta / total_mass
	x  = x + tau * x_dot
	x_dot = x_dot + tau * xacc
	theta = theta + tau * theta_dot
	theta_dot = theta_dot + tau * thetaacc
	
	done =  x < -x_threshold \
			or x > x_threshold \
			or theta < -theta_threshold_radians \
			or theta > theta_threshold_radians
	return done

class QLearningCausal(QLearning):
	"""docstring for QLearningCausal"""
	def epsilon_greedy(self, state, discretized_state):
		eps = np.random.uniform()
		if eps > 0.9:
			return np.argmax(self.Q[discretized_state]), None
		if eps < 0.000005:
			if not counterfactual(state, 0):
				return 0, None
			if not counterfactual(state, 1):
				return 1, None
		return self.env.action_space.sample(), None
		# return np.argmax(best), None
def main():
	import time
	total = []
	# for i in range(10):
	# 	time_s = time.time()
	q = QLearningCausal()
	avg = q.train()
	total.append(avg)
		# q.train()
	mean_reward = np.mean(total, axis=0)
	for i in mean_reward:
		print(i)
	q.test()

if __name__ == '__main__':
	main()