import numpy as np

from PIL import Image
import cv2

from rl.policy import EpsGreedyQPolicy

def convert_arr(array, num):
    five_switches = [2, 6, 4, 8, 0]
    seven_switches = [4, 2, 0, 1, 8, 6, 7]
    nine_switches = [5, 3, 4, 2, 0, 1, 8, 6, 7]
    ans = [0 for _ in range(num)]
    for i in range(num):
        if num == 5:
            if array[five_switches[i]] == 1: ans[i] = 1
        if num == 7:
            if array[seven_switches[i]] == 1: ans[i] = 1
        if num == 9:
            if array[nine_switches[i]] == 1: ans[i] = 1
    return ans
class Policy(object):
    def __init__(self, eps_max, eps_min, eps_test, nb_steps, env, causal=False, ratio=0.5):
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_test = eps_test
        self.nb_steps = nb_steps
        self.env = env
        self.step = 0
        self.causal = causal
        self.use_model = False
        self.current_eps = eps_max

    def get_current_value(self, training=True):
        if training:
            a = -float(self.eps_max - self.eps_min) / float(self.nb_steps)
            b = float(self.eps_max)
            value = max(self.eps_min, a * float(self.step) + b)
        else:
            value = self.eps_test
        # self.step = (self.step +  1) % self.nb_steps
        self.step += 1
        # print("eps = {}".format(value))
        self.current_eps = value
        return value
    def select_action(self, state, Q, training=True):
        raise NotImplementedError

class EpsilonGreedy(Policy):
    def select_action(self, state, Q, training=True, prob_explore=0.0):
        r = np.random.uniform()
        eps = self.get_current_value(training)
        if r > eps:
            self.use_model = False
            return np.argmax(Q[state])
        r = np.random.uniform()
        if not self.causal:
            return self.env.sample_action()
        goal = self.env.get_goal()
        macro_state = self.env.get_state()
        targets = []
        ##Here we have to add the masterswitch checking function
        for i in range(len(goal)):
            if goal[i] != macro_state[i]:
                targets.append(i + self.env.num)
        np.random.shuffle(targets)
        for target in targets:
            actions = self.env.causal_structure.get_causes(target)
            if len(actions) > 0:
                if self.env.structure == "masterswitch" and self.env.get_switches_state()[self.env.get_masterswitch()] != 1:
                    self.use_model = True
                    return self.env.get_masterswitch()
                self.use_model = True
                return actions.pop()  
        self.use_model = False
        return self.env.sample_action()

class EpsilonGreedyDQN(EpsGreedyQPolicy):
    def __init__(self, env, num, model=None, causal=False):
        super().__init__()
        self.env = env
        self.use_causal_info = causal
        self.model = model
        self.num = num
        self.counter = 0
    def select_action(self, q_values, observation):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        r = np.random.uniform()
        if r > self.eps:
            return np.argmax(q_values)
        if not self.use_causal_info:
            return self.env.sample_action()
        r = np.random.uniform()
        if r > 0.9:
            return self.env.sample_action()
        goal = self.env.get_goal()
        lights_on = self.model.predict(observation.reshape(1, 84, 84, 1))
        macro_state = np.rint(lights_on[0]).astype(int)
        macro_state = convert_arr(macro_state, self.num)
        targets = []
        for i in range(len(goal)):
            if goal[i] != macro_state[i]:
                targets.append(i + self.env.num)
        np.random.shuffle(targets)
        for target in targets:
            actions = self.env.causal_structure.get_causes(target)
            if len(actions) > 0:
                self.counter += 1
                return actions.pop()      
        return self.env.sample_action()
