import os
import random
import time
import collections

import numpy as np
import gym


from simple_dqn import DQNAgent
from causal_dqn import CausalDQN

if __name__ == "__main__":
    env = gym.make('{}-v{}'.format("CartPole", 1))
    episodes = 100
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, weights_file='cartpole{}.h5'.format(episodes))
    causal_agent = CausalDQN(state_size, action_size, weights_file='causal_cartpole{}.h5'.format(episodes))
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
            # if e > 0 and (e + 1) % 1 == 0:
            rewards_dqn.append(step)
            print("Episode: {} / {}, Score:  {}".format(e + 1, episodes, step))
            agent.replay(32)
    finally:
        agent.save_model()
    print("=" * 50)
    try:
        reward_causal_dqn = []
        for e in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            done = False
            step = 0
            while not done:
                step += 1
                action = causal_agent.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                causal_agent.remember(state, action, reward, next_state, done)
                state = next_state
            # if e > 0 and (e + 1) % 1 == 0:
            reward_causal_dqn.append(step)
            print("Episode: {} / {}, Score:  {}".format(e + 1, episodes, step))
            causal_agent.replay(32)
    finally:
        causal_agent.save_model()
    env.close()