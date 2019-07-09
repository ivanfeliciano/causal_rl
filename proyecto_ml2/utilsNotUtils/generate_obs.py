import gym
import time
import math
env = gym.make("CartPole-v1")

state = env.reset()
# pred = [set() for i in range(500)]
# rewards = [[[0 for i in range(6)] for j in range(500)] for k in range(500)]
# model = [[None for i in range(6)] for j in range(500)]
print("state = {}".format(state))
done = False
while not done:
	time.sleep(0.1)
	env.render()
	# theta_threshold_radians = 12 * 2 * math.pi / 360
	# x_threshold = 2.4
	# x, x_dot, theta, theta_dot = state
	action = env.action_space.sample() # your agent here (this takes random actions)
	print("action = {}".format("left" if action == 0 else "right"))
	state, reward, done, info = env.step(action)
	print("state = {}".format(state))

	# print(env.step(action))

	# gravity = 9.8
	# masscart = 1.0
	# masspole = 0.1
	# total_mass = (masspole + masscart)
	# length = 0.5 # actually half the pole's length
	# polemass_length = (masspole * length)
	# force_mag = 10.0
	# tau = 0.02  # seconds between state updates

	# force = force_mag if action==1 else -force_mag
	# costheta = math.cos(theta)
	# sintheta = math.sin(theta)
	# temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
	# thetaacc = (gravity * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
	# xacc  = temp - polemass_length * thetaacc * costheta / total_mass
	# x  = x + tau * x_dot
 #    x_dot = x_dot + tau * xacc
 #    theta = theta + tau * theta_dot
	# theta_dot = theta_dot + tau * thetaacc
	# done =  x < -x_threshold \
	# 		or x > x_threshold \
	# 		or theta < -theta_threshold_radians \
	# 		or theta > theta_threshold_radians

	if (done):
		# print("Pole fall if move to {}".format("left" if action == 0 else "right"))
		state = env.reset()
env.close()

#   model[state][action] = (reward, new_state)
#   rewards[state][new_state][action] = reward
#   pred[new_state].add((state, action))
# if done:
#   state = env.reset()

# print("pred = {}".format(pred))
# print("rewards = {}".format(rewards))
# print("model = {}".format(model))
