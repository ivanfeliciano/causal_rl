import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(x_axis, mean_vecs, std_dev_vectors, labels, filename):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Episodios')
    ax1.set_ylabel('Recompensa promedio')
    for i in range(len(mean_vecs)):
        plt.plot(x_axis, mean_vecs[i], label=labels[i])
        plt.fill_between(x_axis, mean_vecs[i] - std_dev_vectors[i], mean_vecs[i] + std_dev_vectors[i],\
                        alpha=0.2)
    plt.legend(loc='best')
    plt.savefig("{}.pdf".format(filename), bbox_inches='tight')