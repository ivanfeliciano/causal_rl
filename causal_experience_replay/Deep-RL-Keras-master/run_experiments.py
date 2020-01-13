""" Deep RL Algorithms for OpenAI Gym environments
"""

import os
import sys
import gym
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from A2C.a2c import A2C
from A3C.a3c import A3C
from DDQN.ddqn import DDQN
from DDPG.ddpg import DDPG
from DDQNCausal.ddqn_causal import DDQNCausal

from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical

from utils.atari_environment import AtariEnvironment
from utils.continuous_environments import Environment
from utils.networks import get_session

gym.logger.set_level(40)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--type', type=str, default='DDQN',help="Algorithm to train from {A2C, A3C, DDQN, DDPG, DDQNCausal}")
    parser.add_argument('--with_PER', dest='with_per', action='store_true', help="Use Prioritized Experience Replay (DDQN + PER)")
    parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")
    #
    parser.add_argument('--nb_episodes', type=int, default=1000, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--consecutive_frames', type=int, default=4, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--training_interval', type=int, default=30, help="Network training frequency")
    parser.add_argument('--n_threads', type=int, default=8, help="Number of threads (A3C)")
    #
    parser.add_argument('--gather_stats', dest='gather_stats', action='store_true',help="Compute Average reward per episode (slower)")
    parser.add_argument('--render', dest='render', action='store_true', help="Render environment while training")
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4',help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    #
    parser.set_defaults(render=False)

    parser.add_argument('--stochastic', dest='stochastic', action='store_true', help='Choose stochastic env')
    return parser.parse_args(args)

def main(args=None):

    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Check if a GPU ID was set
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    number_of_experiments = 5
    for i in range(number_of_experiments):
        set_session(get_session())
        summary_writer = tf.summary.FileWriter(args.type + "/tensorboard_" + args.env)
        env = Environment(gym.make(args.env), args.consecutive_frames)
        state_dim = env.get_state_size()
        action_dim = gym.make(args.env).action_space.n
        
        #DQN
        try:
            print("DQN {}".format(i + 1))
            env.reset()
            dqn = DDQN(action_dim, state_dim, args)
            stats = dqn.train(env, args, summary_writer)
            df = pd.DataFrame(np.array(stats))
            df.to_csv("allLogs/logsDQN{}_{}.csv".format(i + 1, number_of_experiments), header=['Episode', 'Reward', 'std'], float_format='%10.5f')
        except:
            print("DQN failed")
        #DQN + PER
        try:
            print("DQN PER{}".format(i + 1))
            env.reset()
            args.with_per = True
            dqn = DDQN(action_dim, state_dim, args)
            stats = dqn.train(env, args, summary_writer)
            df = pd.DataFrame(np.array(stats))
            df.to_csv("allLogs/logsDQN_PER{}_{}.csv".format(i + 1, number_of_experiments), header=['Episode', 'Reward', 'std'], float_format='%10.5f')
            args.with_per = False
        except:
            print("DQN PER failed")

        #DQN + CM
        try:
            print("DQN CM{}".format(i + 1))
            env.reset()
            dqn = DDQNCausal(action_dim, state_dim, args)
            stats = dqn.train(env, args, summary_writer)
            df = pd.DataFrame(np.array(stats))
            df.to_csv("allLogs/logsDQN_CM{}_{}.csv".format(i + 1, number_of_experiments), header=['Episode', 'Reward', 'std'], float_format='%10.5f')
        except:
            print("DQN CM failed")
        #DQN + PER + CM
        # try:
        #     env.reset()
        #     args.with_per = True
        #     dqn = DDQNCausal(action_dim, state_dim, args)
        #     stats = dqn.train(env, args, summary_writer)
        #     df = pd.DataFrame(np.array(stats))
        #     df.to_csv("allLogs/logsDQN_CM_PER{}_{}.csv".format(i + 1, number_of_experiments), header=['Episode', 'Reward', 'std'], float_format='%10.5f')
        #     args.with_per = False
        # except:
        #     print("DQN PER CM failed")
        # Train
        env.env.close()

if __name__ == "__main__":
    main()
