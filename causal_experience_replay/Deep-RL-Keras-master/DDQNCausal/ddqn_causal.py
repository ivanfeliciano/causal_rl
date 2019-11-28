import random
import math
import numpy as np
from random import random, randrange

from DDQN.ddqn import DDQN

class DDQNCausal(DDQN):
    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action
        """
        if random() <= self.epsilon:
            action = randrange(self.action_dim)
            cart_pos = s[-1][0]
            pole_angle = s[-1][2]
            # Prueba izquierda 0
            new_cart_pos = cart_pos - 0.01
            new_angle = cart_pos + math.radians(1)
            z = True if abs(new_cart_pos) < abs(cart_pos) and abs(new_angle) < abs(pole_angle) else False
            if z: return 0
            # Prueba derecha 1
            new_cart_pos = cart_pos + 0.01
            new_angle = cart_pos + math.radians(-1)
            z = True if abs(new_cart_pos) < abs(cart_pos) and abs(new_angle) < abs(pole_angle) else False
            if z: return 1
            return action
        else:
            return np.argmax(self.agent.predict(s)[0])