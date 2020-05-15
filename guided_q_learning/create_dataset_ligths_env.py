import os
import sys
import csv

import numpy as np 
import cv2

from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import mujoco_py

from utils.lights_env_helper import powerset

five_switches = [2, 6, 4, 8, 0]
seven_switches = [4, 2, 0, 1, 8, 6, 7]
nine_switches = [5, 3, 4, 2, 0, 1, 8, 6, 7]
visited = dict()

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    csv_file = sys.argv[2] if len(sys.argv) > 2 else "./data/train.csv"
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["filename"] + ["Zone {}".format(_) for _ in range(9)])
        for i in [5, 7, 9]:
            fullpath = os.path.join(os.path.dirname(__file__), 'env/assets', "arena_v2_" + str(i) + ".xml")
            if not os.path.exists(fullpath):
                raise IOError('File {} does not exist'.format(fullpath))
            model = mujoco_py.load_model_from_path(fullpath)
            sim = mujoco_py.MjSim(model)
            for config in powerset(i):
                sim.model.light_active[:] = np.array(config)
                label = [0 for _ in range(9)]
                for k in range(len(config)):
                    if config[k] == 1 and i == 5 : label[five_switches[k]] = 1
                    if config[k] == 1 and i == 7 : label[seven_switches[k]] = 1
                    if config[k] == 1 and i == 9 : label[nine_switches[k]] = 1
                print(config, label)
                img = sim.render(width=32, height=32, camera_name="birdview")
                im_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                filename = "".join([str(_) for _ in label])
                if not visited.get(filename):
                    cv2.imwrite("./data/images/{}.png".format(filename), im_rgb)
                    csv_writer.writerow(["'" + filename + "'"] + label)
                    visited[filename] = True
