import sys

import numpy as np
import cv2

from env.light_env import LightEnv

n_switches = int(sys.argv[1]) if len(sys.argv) > 1 else 5
structure = sys.argv[2] if len(sys.argv) > 2 else "one_to_one"
env = LightEnv(horizon=n_switches, num=n_switches, structure=structure)

print(n_switches, structure)

env.keep_struct = False
env.reset()
env.keep_struct = True
goal_image = env.goalim
goal_image = cv2.cvtColor(goal_image.astype('float32'), cv2.COLOR_RGB2GRAY)
goal_image[goal_image >= .5] = 1.
goal_image[goal_image < .5] = 0.
obs = env._get_obs(images=True)
print(env.aj)
while True:
    obs = cv2.cvtColor(obs.astype('float32'), cv2.COLOR_RGB2GRAY)
    obs[obs >= .5] = 1.
    obs[obs < .5] = 0.
    image = np.hstack((goal_image, obs))
    cv2.imshow('Rooms', image)
    k = cv2.waitKey()
    if k == ord('q'): break
    action = int(k) - ord('0')
    print(action)
    state, r, done, info = env.step(action)
    print("State : {}".format(state[:n_switches]))
    print("Reward : {}".format(r))
    obs = env._get_obs(images=True)
cv2.destroyAllWindows()



# self.sim.model.light_active[:] = light
#         img = self.sim.render(width=500,height=500,camera_name="birdview")
#         # im_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         # self.index += 1
#         # cv2.imwrite("test{}.png".format(self.index), im_rgb)