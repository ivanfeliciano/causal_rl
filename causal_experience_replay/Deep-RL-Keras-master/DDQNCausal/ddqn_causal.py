import random
import math
import numpy as np
from random import random, randrange

from DDQN.ddqn import DDQN

class DDQNCausal(DDQN):
    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action
        """
        pole_angle = s[-1][2]
        # test to the left
        new_angle = pole_angle + math.radians(4)
        z = True if abs(new_angle) < abs(pole_angle) else False
        if z:
            return 0
        # test to the right
        new_angle = pole_angle + math.radians(-4)
        z = True if abs(new_angle) < abs(pole_angle) else False
        if z:
            return 1
        if random() <= self.epsilon:
            return randrange(self.action_dim)
        else:
            return np.argmax(self.agent.predict(s)[0])