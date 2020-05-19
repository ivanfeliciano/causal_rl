import numpy as np
import cv2
from PIL import Image
from rl.core import Processor

class LightEnvProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3
        observation = cv2.cvtColor(observation.astype('float32'), cv2.COLOR_RGB2GRAY) / 255
        observation[observation >= .5] = 1.
        observation[observation < .5] = 0.
        # img = Image.fromarray(observation)
        # img = img.convert('L')
        # return np.array(img).astype('uint8')
        return observation
    # def process_state_batch(self, batch):
    #     return batch.astype('float32')