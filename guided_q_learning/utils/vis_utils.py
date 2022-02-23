import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

def plot_rewards(x_axis, mean_vecs, std_dev_vectors, labels, filename, log=False, filetype="pdf"):
    fig, ax1 = plt.subplots()
    if log:
        plt.yscale("log")
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average reward')
    for i in range(len(mean_vecs)):
        plt.plot(x_axis, mean_vecs[i], label=labels[i])
        plt.fill_between(x_axis, mean_vecs[i] - std_dev_vectors[i], mean_vecs[i] + std_dev_vectors[i],\
                        alpha=0.2)
    plt.legend(loc='best')
    plt.savefig("{}.{}".format(filename, filetype), bbox_inches='tight')
    plt.close()

def plot_boxplot(df, col_x, col_y, hue, filename):
    fig, ax = plt.subplots()
    sns.boxplot(x=col_x, y=col_y, hue=hue, data=df, palette="tab10", ax=ax)
    plt.savefig(f"{filename}.png")
    plt.close()