import numpy as np
from scipy import stats


def read_mat_from_file(filepath):
    return np.load(filepath, allow_pickle=True)

def compute_mean_and_std_dev(rewards_np_mats):
    n_algorithms = rewards_np_mats.shape[0]
    mean_vectors = [_ for _ in range(n_algorithms)]
    std_dev_vectors = [_ for _ in range(n_algorithms)]
    for i in range(n_algorithms):
        mean_vectors[i] = np.mean(rewards_np_mats[i], axis=0)
        std_dev_vectors[i] = np.std(rewards_np_mats[i], axis=0)
    return mean_vectors, std_dev_vectors

def transform_to_modulated_matrix(rewards_np_mats, mod):
    mod_matrix = np.empty(shape=rewards_np_mats.shape, dtype='object')
    for i in range(rewards_np_mats.shape[0]):
        for j in range(rewards_np_mats.shape[1]):
            new_size = rewards_np_mats[i, j].shape[0] // mod
            holder = np.zeros(new_size)
            for k in range(holder.shape[0]):
                holder[k] = np.mean(rewards_np_mats[i, j][k * mod: (k + 1) * mod]) 
            mod_matrix[i, j] = holder
    return mod_matrix

def compute_stat_test(mean_vectors, number_of_evaluation_episodes=100, alpha=0.05):
    mean_rewar_eval = []
    p_values_welch = []
    for i in range(len(mean_vectors)):
        mean = np.mean(mean_vectors[i][-number_of_evaluation_episodes:])
        std = np.std(mean_vectors[i][-number_of_evaluation_episodes:])
        mean_rewar_eval.append((mean, std))
        mean_q = np.mean(mean_vectors[0][-number_of_evaluation_episodes:])
        std_q = np.std(mean_vectors[0][-number_of_evaluation_episodes:])
        stat, p = stats.ttest_ind(mean_vectors[0][-number_of_evaluation_episodes:], mean_vectors[i][-number_of_evaluation_episodes:], equal_var=False)
        p_values_welch.append(p)
    return mean_rewar_eval, p_values_welch

def array_to_html_table(header, array):
    table = "<table>\n"
    table += "  <tr>\n"
    for column in header:
        table += "    <th>{0}</th>\n".format(column.strip())
    table += "  </tr>\n"
    for row in array:
        table += "  <tr>\n"
        for column in row:
            if isinstance(column, float) or isinstance(column, float):
                table += "    <td>{:.2}</td>\n".format(column)
            else:
                table += "    <td>{}</td>\n".format(column.strip())
        table += "  </tr>\n"
    table += "</table>"
    return table

