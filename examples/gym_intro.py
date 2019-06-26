# -*- coding: utf-8 -*-
import random
import os
import sys
import time

import numpy as np
import gym

from gym.envs.registration import register
# register(
#     id='Deterministic-4x4-FrozenLake-v0',
#     entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
#     kwargs={'map_name': '4x4', 'is_slippery': False}
# )


# Hiperparámetros
NUMBER_OF_EPISODES = 100000
ALPHA_LR = 0.8
GAMMA_DF = 0.95
EPSILON = 0.2

# Inicialización de Gymai
env = gym.make('Taxi-v2')
from gym.envs.registration import register
number_of_actions = env.action_space.n
number_of_states = env.observation_space.n
print("# Action = {}".format(number_of_actions))
print("# States = {}".format(number_of_states))
Q = np.zeros([number_of_states, number_of_actions])
print(env.observation_space)

state = env.reset()
env.render()
action = env.action_space.sample()
# print(action)
new_state, reward, done, info = env.step(action)
state = env.decode(new_state)
print(list(state))
env.render()
sys.exit()

for episode in range(NUMBER_OF_EPISODES):
    state = env.reset()
    # print("Episode {}".format(episode))
    done = False
    while not done:
        # Elige la mejor acción
        if random.random() > EPSILON:
            action = np.argmax(Q[state, :])
        # Elige una acción aleatoria
        else:
            action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        if reward == 1:
            reward = 1
            Q[new_state] = np.ones(number_of_actions) * reward
        else:
            if not done:
                reward = 0
            else:
                reward = 0
                Q[new_state] = np.ones(number_of_actions) * reward
        Q[state, action] = Q[state, action] + ALPHA_LR * (reward + GAMMA_DF * np.max(Q[new_state, :]) - Q[state, action])
        state = new_state
    # os.system('clear')
    # env.render()
print("Final Q")
print(Q)

print("Shortest path")
state = env.reset()
env.render()
done = False

while not done:
  action = np.argmax(Q[state])
  new_state, reward, done, info = env.step(action)
  print("+++++++++++++++++++++++")
  # os.system('clear')
  env.render()
  state = new_state

env.close()


