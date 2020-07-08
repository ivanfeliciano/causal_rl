import sys

import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model


from env.light_env import LightEnv



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


n_switches = int(sys.argv[1]) if len(sys.argv) > 1 else 5
structure = sys.argv[2] if len(sys.argv) > 2 else "one_to_one"
env = LightEnv(horizon=n_switches, num=n_switches, structure=structure)

print(n_switches, structure)
model = load_model('models/multilabel_classifier84.h5')

env.keep_struct = False
env.reset()
env.keep_struct = True
goal_image = env.goalim
goal_image = cv2.cvtColor(goal_image.astype('float32'), cv2.COLOR_BGR2RGB)
# goal_image[goal_image >= .5] = 1.
# goal_image[goal_image < .5] = 0.
obs = env._get_obs(images=True)[1]
print(env.aj)
print("State : {}".format(env._get_obs()[0][:n_switches]))
while True:    
    obs = cv2.cvtColor(obs.astype('float32'), cv2.COLOR_BGR2RGB) / 255
    # obs[obs >= .5] = 1.
    # obs[obs < .5] = 0.
    # # lights_on = model.predict(obs.reshape(1, 84, 84, 1))
    # # macro_state = np.rint(lights_on[0]).astype(int)
    # # print("Predicted : {}".format(macro_state))
    # # macro_state = convert_arr(macro_state, n_switches)
    # print("Conveted pred : {}".format(macro_state))
    space = np.zeros((500, 100, 3)) + 255
    image = np.hstack((obs, space, goal_image))
    cv2.imshow('Rooms', image)
    k = cv2.waitKey()
    if k == ord('q'): break
    action = int(k) - ord('0')
    print(action)
    state, r, done, info = env.step(action)
    print("State : {}".format(state[0][:n_switches]))
    print("Reward : {}".format(r))
    obs = env._get_obs(images=True)[1]
cv2.destroyAllWindows()



# self.sim.model.light_active[:] = light
#         img = self.sim.render(width=500,height=500,camera_name="birdview")
#         # im_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         # self.index += 1
#         # cv2.imwrite("test{}.png".format(self.index), im_rgb)