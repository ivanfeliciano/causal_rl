# -*- coding: utf-8 -*-
import warnings
import random
import os
import time
from copy import deepcopy

import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History

from rl.callbacks import (
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer
)



from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from env.light_env import LightEnv
from environments import LightAndSwitchEnv
from policy import EpsilonGreedy, EpsilonGreedyDQN, convert_arr
from utils.lights_env_helper import aj_to_adj_list

from processor import LightEnvProcessor

class QLearningAgent(object):
    """
    Clase base para un agente de Q learning.
    Aquí se configura el ambiente, los parámetros y
    el flujo del aprendizaje.
    """
    def __init__(self, environment, policy, causal=False, episodes=100, alpha=0.8, gamma=0.95, mod_episode=1):
        self.env = environment
        self.policy = policy
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.Q = self.env.init_q_table()
        self.mod_episode = mod_episode
        self.training = True
        self.causal = causal
    def select_action(self, state):
        if self.training:
            return self.policy.select_action(state, self.Q)
        return self.policy.select_action(state, self.Q, False)
    def train(self):
        self.training = True
        self.avg_reward = []
        self.avg_eps = []
        self.avg_queries = []
        rewards_per_episode = []
        for episode in range(self.episodes):
            assisted_times_per_episode = []
            epsilones = []
            total_episode_reward = 0
            state = self.env.reset()
            done = False
            step = 0
            assisted_times = 0
            while not done:
                action = self.select_action(state)
                new_state, reward, done, info = self.env.step(action)
                self.Q[state][action] = self.Q[state][action] + self.alpha * \
                                        (reward + self.gamma * np.max(self.Q[new_state]) -\
                                        self.Q[state][action])
                state = new_state
                total_episode_reward += reward
                step += 1
                if self.policy.causal and self.policy.use_model:
                    assisted_times += 1
            rewards_per_episode.append(total_episode_reward)
            epsilones.append(self.policy.current_eps)
            assisted_times_per_episode.append((assisted_times / step) * 100)
            # self.policy.step = 0
            if episode == 0 or (episode + 1) % self.mod_episode == 0:
                self.avg_eps.append(np.mean(epsilones))
                self.avg_queries.append(np.mean(assisted_times_per_episode))
                rewards_per_episode = self.update_avg_reward(rewards_per_episode)
        return self.avg_reward
    def update_avg_reward(self, rewards_per_episode):
        episodes_block_avg_rwd = np.mean(rewards_per_episode)
        self.avg_reward.append(episodes_block_avg_rwd)
        return []
    def test(self):
        self.training = False
        rewards_per_episode = []
        self.avg_reward = []
        for episode in range(self.episodes):
            total_episode_reward = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                new_state, reward, done, info = self.env.step(action)
                state = new_state
                total_episode_reward += reward
            rewards_per_episode.append(total_episode_reward)
            if episode == 0 or  (episode + 1) % self.mod_episode == 0:
                rewards_per_episode = self.update_avg_reward(rewards_per_episode)
        return self.avg_reward
    def plot_avg_reward(self, filename="average_reward"):
        x_axis = self.mod_episode * (np.arange(len(self.avg_reward)))
        plt.plot(x_axis, self.avg_reward)
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.savefig("plots/{}.png".format(filename))     
        plt.close()
    def get_training_avg_reward(self):
        return self.avg_reward


class AssistedDQN(DQNAgent):
    def fit(self, env, nb_steps=0, episodes=100, action_repetition=0, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None, mod_episode=1):
        if not self.compiled:
            raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        self.training = True
        rewards_per_episode = []
        avg_reward = []
        callbacks = [] if not callbacks else callbacks[:]
        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()
        episode = np.int16(0)
        self.step = np.int16(0)
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while episode < episodes:
                if observation is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = np.int16(0)
                    episode_reward = np.float32(0)
                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    assert observation is not None
                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None
                # Run a single step.
                callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = np.float32(0)
                accumulated_info = {}
                done = False
                callbacks.on_action_begin(action)
                observation, r, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(observation, r, done, info)
                    # print(r, done, info)
                callbacks.on_action_end(action)
                reward += r
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                metrics = self.backward(reward, terminal=done)
                episode_reward += reward
                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    self.forward(observation)
                    self.backward(0., terminal=False)
                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)
                    rewards_per_episode.append(episode_reward)
                    if episode == 0 or (episode + 1) % mod_episode == 0:
                        avg_reward.append(np.mean(rewards_per_episode))
                        rewards_per_episode = []
                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()
        return avg_reward
    def get_training_avg_reward(self):
        return self.avg_reward
    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        q_values = self.compute_q_values(state)
        if self.training:
            action = self.policy.select_action(q_values=q_values, observation=observation)
        else:
            action = self.test_policy.select_action(q_values=q_values, observation=observation)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action


class DQN(object):
    def __init__(self, environment, nb_actions, processor, policy, episodes=1000, mod_episode=100):
        self.env = environment
        self.episodes = episodes
        self.mod_episode = mod_episode
        self.nb_actions = nb_actions
        self.model = self.init_model()
        self.memory = SequentialMemory(limit=1000000, window_length=1)
        self.processor = processor
        self.policy = policy 
        self.dqn_agent = AssistedDQN(model=self.model, nb_actions=self.nb_actions,\
                                policy=self.policy, memory=self.memory, processor=self.processor,\
                                nb_steps_warmup=self.nb_actions, gamma=0.95, target_model_update=1, delta_clip=1.)
        self.dqn_agent.compile(Adam(lr=0.00025), metrics=['mae'])
        self.get_training_avg_reward = []
    def get_training_avg_reward(self):
        return self.avg_reward
    def plot_avg_reward(self, filename="average_reward_dqn"):
        x_axis = self.mod_episode * (np.arange(len(self.avg_reward)))
        plt.plot(x_axis, self.avg_reward)
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.savefig("plots/{}.png".format(filename))     
        plt.close()
    def init_model(self):
        model = Sequential()
        model.add(Permute((2, 3, 1), input_shape=(1 , 84, 84)))
        model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.nb_actions))
        model.add(Activation('linear'))
        return model
    def train(self, filename="dqn_lights"):
        weights_filename = 'models/{}_weights.h5f'.format(filename)
        checkpoint_weights_filename = 'models/' + filename + '_weights_{step}.h5f'
        log_filename = 'logs/{}_log.json'.format(filename)
        # callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=self.mod_episode)]
        callbacks = [FileLogger(log_filename, interval=self.mod_episode)]
        self.avg_reward = self.dqn_agent.fit(self.env, callbacks=callbacks, action_repetition=0, episodes=self.episodes, log_interval=self.mod_episode, mod_episode=self.mod_episode,
        nb_max_episode_steps=self.env.horizon)
        # self.dqn_agent.save_weights(weights_filename, overwrite=True)
        print("Steps {}".format(self.dqn_agent.step))
        print("Times Used CM {}".format(self.policy.inner_policy.counter))
        # self.plot_avg_reward()
        return self.avg_reward
    def test(self, filename="dqn_lights"):
        weights_filename = 'models/{}_weights.h5f'.format(filename)
        self.dqn_agent.load_weights(weights_filename)
        self.dqn_agent.test(self.env, nb_episodes=10)


def discrete():
    horizon = 5
    num = 5
    structure = "one_to_one"
    env = LightEnv(horizon=horizon, num=num, structure=structure)
    env.keep_struct = False
    env.reset()
    env.keep_struct = True
    full_adj_list = aj_to_adj_list(env.aj)
    vanilla_env = LightAndSwitchEnv(env, full_adj_list)
    
    eps_policy = EpsilonGreedy(1, 0.1, 0.1, horizon, vanilla_env, True)
    q = QLearningAgent(vanilla_env, eps_policy, mod_episode=20)
    average_reward = q.train()
    q.plot_avg_reward("basis_experiment_causal")

def continuos():
    horizon = 5
    num = 5
    structure = "one_to_one"
    env = LightEnv(horizon=horizon, num=num, structure=structure)
    env.keep_struct = False
    env.reset()
    env.keep_struct = True
    full_adj_list = aj_to_adj_list(env.aj)
    simple_env = LightAndSwitchEnv(env=env, adj_list=full_adj_list, discrete=False)
    nb_actions = num + 1
    processor = LightEnvProcessor()
    episodes = 1000
    mod_episode = 50
    model = load_model('models/multilabel_classifier84.h5')
    policy = LinearAnnealedPolicy(EpsilonGreedyDQN(simple_env, num, model, False),\
                                            attr='eps', value_max=1., value_min=.1,\
                                            value_test=.05, nb_steps=horizon)
    agent = DQN(simple_env, nb_actions, processor, policy, episodes, mod_episode=mod_episode)
    agent.train()
    
if __name__ == '__main__':
    continuos()

