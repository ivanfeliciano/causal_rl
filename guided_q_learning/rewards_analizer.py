import sys
from os import listdir
from os.path import isfile, join

import numpy as np

from utils.performance_utils import *
from utils.vis_utils import plot_rewards

filename = "/home/ivan/Documentos/causal_rl/guided_q_learning/results_thesis/light_switches/discrete/experiment_delta/rewards_mats/deterministic_low_0.25_many_to_one_N_5_experiments_10_episodes_5000_eps_6250.npy"

input_directory = sys.argv[1]
output_file_name = sys.argv[2]
experiment_name = sys.argv[3]
base_dir_plots = sys.argv[4]
mod = int(sys.argv[5])
n_eval_episodes = int(sys.argv[6])
fileout = open(output_file_name, "w")
alpha = 0.05
html_text = "<h1>{}</h1>".format(experiment_name)


onlyfiles = sorted([f for f in listdir(input_directory) if isfile(join(input_directory, f))])
labels = ["Q-learning", "Q-learning + estructura completa", \
            "Q-learning + estructura parcial", "Q-learning + estructura incorrecta"]
for file_path in onlyfiles:
	name = file_path[:-4]
	splited_name = name.strip().split("_")
	delta = float(splited_name[-1]) / (float(splited_name[11]) * float(splited_name[7]))
	html_text += "<h2>{} delta {:.2f}</h2>".format(" ".join(name.strip().split("_")), delta)
	rewards = read_mat_from_file(join(input_directory, file_path))
	# primero grafico
	mod_mat = transform_to_modulated_matrix(rewards, mod)
	mean_vectors, std_dev_vectors = compute_mean_and_std_dev(mod_mat)
	x_axis = mod * (np.arange(len(mean_vectors[0])))
	plot_path = join(base_dir_plots, name)
	plot_rewards(x_axis, mean_vectors, std_dev_vectors, labels, plot_path)
	# html_text += "<iframe src='{}.pdf'' width='50%' height='500px'></iframe>".format(plot_path)
	html_text += '<object data="{}.pdf" type="application/pdf" width="700px" height="700px"><embed src="{}.pdf"><p>This browser does not support PDFs. Please download the PDF to view it: <a href="{}.pdf">Download PDF</a>.</p></embed></object>'.format(plot_path, plot_path, plot_path)
	mean_vectors, std_dev_vectors = compute_mean_and_std_dev(rewards)
	mean_eval, pvalues = compute_stat_test(mean_vectors, n_eval_episodes)

	array_to_table_means = []
	array_to_table_pvalues = []
	for label, value in zip(labels, mean_eval):
		mean_std = "${:.4f} \\pm {:.4f}$".format(value[0], value[1])
		array_to_table_means.append((label, mean_std))
	for label, value in zip(labels, pvalues):
		is_rejected = "rejected" if value < alpha else "fail to reject"
		array_to_table_pvalues.append((label, "${:.4f}$".format(value), is_rejected))
	header = ["Algoritmo", "Recompensa promedio E = {}".format(n_eval_episodes)]
	html_text += array_to_html_table(header, array_to_table_means)
	header = ["Algoritmo", "pvalue vs Q-learning"]
	html_text += array_to_html_table(header, array_to_table_pvalues)
fileout.writelines(html_text)
fileout.close()
#leer archivos de carpeta
