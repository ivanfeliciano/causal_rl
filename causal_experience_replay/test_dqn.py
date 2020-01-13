import time
from simple_dqn import DQNAgent
from causal_dqn import CausalDQN
import gym
import numpy as np

if __name__ == "__main__":
    n_tests = 1
    env = gym.make('{}-v{}'.format("CartPole", 1))
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = CausalDQN(state_size, action_size, weights_file="causal_cartpole1000.h5")
    for i in range(n_tests):
        state = env.reset()
        done = False
        step = 0
        while not done:
            step += 1
            time.sleep(0.1)
            env.render()
            state = np.reshape(state, [1, state_size])
            action = agent.predict(state)
            state, reward, done, _ = env.step(action)
        print(step)
            
                