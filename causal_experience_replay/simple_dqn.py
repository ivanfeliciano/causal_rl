import os
import random
import time
import collections

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import gym


class DQNAgent(object):
    def __init__(self, state_size, action_size, weights_file="weights.h5"):
        self.weight_backup = weights_file
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=10000)
        self.gamma = 0.95
        self.eps = 1.0
        self.flag = False
        self.eps_min = 0.01
        self.eps_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()
    def build_model(self):
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.eps = self.eps_min
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.eps:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)[0]
        return np.argmax(act_values)
    def predict(self, state):
        act_values = self.model.predict(state)[0]
        return np.argmax(act_values)
    def replay(self, batch_size):
        if batch_size > len(self.memory):
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            s = time.time()
            self.model.fit(state, target_f, epochs=1, verbose=False)
            if self.eps > self.eps_min:
                self.eps *= self.eps_decay
    def save_model(self):
        self.model.save(self.weight_backup)
if __name__ == "__main__":
    pass
    # env = gym.make('{}-v5'.format("Catcher"))
    # env = gym.make('{}-v{}'.format("CartPole", 1))
    # episodes = 1000
    # state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n
    # agent = DQNAgent(state_size, action_size, weights_file='cartpole{}.h5'.format(episodes))
    # causal_agent = CausalDQN(state_size, action_size, weights_file='causal_cartpole{}.h5'.format(episodes))
    # try:
    #     for e in range(episodes):
    #         state = env.reset()
    #         state = np.reshape(state, [1, state_size])
    #         done = False
    #         step = 0
    #         while not done:
    #             step += 1
    #             action = agent.act(state)
    #             next_state, reward, done, _ = env.step(action)
    #             next_state = np.reshape(next_state, [1, state_size])
    #             agent.remember(state, action, reward, next_state, done)
    #             state = next_state
    #         # if e > 0 and (e + 1) % 1 == 0:
    #         print("Episode: {} / {}, Score:  {}".format(e + 1, episodes, step))
    #         agent.replay(32)
    #     env.close()
    # finally:
    #     agent.save_model()
    # try:
    #     for e in range(episodes):
    #         state = env.reset()
    #         state = np.reshape(state, [1, state_size])
    #         done = False
    #         step = 0
    #         while not done:
    #             step += 1
    #             action = causal_agent.act(state)
    #             next_state, reward, done, _ = env.step(action)
    #             next_state = np.reshape(next_state, [1, state_size])
    #             causal_agent.remember(state, action, reward, next_state, done)
    #             state = next_state
    #         # if e > 0 and (e + 1) % 1 == 0:
    #         print("Episode: {} / {}, Score:  {}".format(e + 1, episodes, step))
    #         causal_agent.replay(32)
    #     env.close()
    # finally:
    #     causal_agent.save_model()