import random
import math
import numpy as np
from simple_dqn import DQNAgent

class CausalDQN(DQNAgent):
    def act(self, state):
        if np.random.rand() <= self.eps:
            action = random.randrange(self.action_size)
            cart_pos = state[0][0]
            pole_angle = state[0][2]
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
        cart_pos = state[0][0]
        pole_angle = state[0][2]
        new_cart_pos = cart_pos - 0.01
        new_angle = cart_pos + math.radians(1)
        z = True if abs(new_cart_pos) < abs(cart_pos) and abs(new_angle) < abs(pole_angle) else False
        if z: return 0
        new_cart_pos = cart_pos + 0.01
        new_angle = cart_pos + math.radians(-1)
        z = True if abs(new_cart_pos) < abs(cart_pos) and abs(new_angle) < abs(pole_angle) else False
        if z: return 1
        act_values = self.model.predict(state)[0]
        return np.argmax(act_values)