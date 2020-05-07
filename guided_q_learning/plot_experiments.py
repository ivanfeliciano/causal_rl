from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt


def plot_rewards(x_axis, mean_vecs, std_dev_vectors, labels, title, filename):
    fig, ax1 = plt.subplots()
    fig.set_size_inches(5, 5)
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average reward')
    ax1.set_title('Average reward')

    for i in range(len(mean_vecs)):
        plt.plot(x_axis, mean_vecs[i], label=labels[i])
        # plt.fill_between(x_axis, mean_vecs[i] - std_dev_vectors[i], mean_vecs[i] + std_dev_vectors[i],\
        #                 alpha=0.2)
    plt.legend(loc='best')
    plt.savefig("{}.png".format(filename))


np_array_files = [f for f in listdir("rewards_data") if isfile(join("rewards_data", f))]

mean_det_rewards = [[None for _ in range(3)] for _ in range(2) ]
std_det_rewards = [[None for _ in range(3)] for _ in range(2) ]
mean_sto_rewards = [[None for _ in range(3)] for _ in range(2) ]
std_sto_rewards = [[None for _ in range(3)] for _ in range(2) ]

for file in np_array_files:
    size, struct, dynamics = file.strip().split('_')
    if size == "7": row = 0
    if size == "9": row = 1
    # if size == "9": row = 2
    if struct == "onetoone": col = 0
    if struct == "onetomany": col = 1
    if struct == "manytoone": col = 2
    rewards = np.load("rewards_data/" + file, allow_pickle=True)
    mean_vectors = [_ for _ in range(4)]
    std_dev_vectors = [_ for _ in range(4)]
    for i in range(4):
        mean_vectors[i] = np.mean(rewards[i], axis=0)
        std_dev_vectors[i] = np.std(rewards[i], axis=0)
    if dynamics[:3] == "det":
        mean_det_rewards[row][col] = mean_vectors
        std_det_rewards[row][col] = std_dev_vectors
    else:
        mean_sto_rewards[row][col] = mean_vectors
        std_sto_rewards[row][col] = std_dev_vectors

mod_episode = 50
fig, axs = plt.subplots(2, 3)
x_axis = mod_episode * (np.arange(len(mean_vectors[0])))
labels = ["Q-learning", "Q-learning estructura completa", \
            "Q-learning estructura parcial", "Q-learning estructura incorrecta"]
for row in range(2):
    for col in range(3):
        mean_vectors = mean_det_rewards[row][col]
        std_dev_vectors = std_det_rewards[row][col]
        for i in range(len(mean_vectors)):
            if row == 0 and col == 0: axs[row, col].set(ylabel="7") 
            if row == 1 and col == 0: axs[row, col].set(ylabel="9")
            # if row == 2 and col == 0: axs[row, col].set(ylabel="9")
            if row == 0 and col == 0: axs[row, col].set_title("Uno a uno", fontsize="x-small") 
            if row == 0 and col == 1: axs[row, col].set_title("Causa común", fontsize="x-small") 
            if row == 0 and col == 2: axs[row, col].set_title("Efecto común", fontsize="x-small") 
            axs[row, col].plot(x_axis, mean_vectors[i], label=labels[i], linewidth=0.75)
            axs[row, col].tick_params(axis='x', labelsize="xx-small")
            axs[row, col].tick_params(axis='y', labelsize="xx-small")

            # axs[row, col].fill_between(x_axis, mean_vectors[i] - std_dev_vectors[i], mean_vectors[i] + std_dev_vectors[i],\
            #             alpha=0.2)
# axs[0, 0].plot(x, y)
# axs[0, 0].set_title('Axis [0,0]')
# axs[0, 1].plot(x, y, 'tab:orange')
# axs[0, 1].set_title('Axis [0,1]')
# axs[1, 0].plot(x, -y, 'tab:green')
# axs[1, 0].set_title('Axis [1,0]')
# axs[1, 1].plot(x, -y, 'tab:red')
# axs[1, 1].set_title('Axis [1,1]')

# for ax in axs.flat:
#     ax.label_outer()

# Hide x labels and tick labels for top plots and y ticks for right plots.
fig.text(0.5, 0.1, 'Episodios', ha='center', va='center', fontsize="small")
fig.text(0.03, 0.5, 'Recompensa promedio', ha='center', va='center', rotation='vertical', fontsize="small")
fig.tight_layout() 
fig.subplots_adjust(bottom=0.17, left=0.12)   ##  Need to play with this number.

fig.legend(labels=labels, loc="lower center", ncol=2, fontsize="x-small")
plt.savefig("rewards_det.pdf")
