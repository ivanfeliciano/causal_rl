import sys
from os import listdir
from os.path import isfile, join

import numpy as np
from statsmodels.stats.power import TTestIndPower

from utils.performance_utils import *
from utils.vis_utils import plot_rewards
from utils.performance_utils import cohend, welch_ttest

def avg_rwd_last_t_episodes(rewards, t=100):
	return np.mean(rewards[:, -t:])# - np.var(rewards[:, -t:])

def max_streaks(rewards, threshold, streak_size):
	streaks = []
	for rwd in rewards:
		max_len = 0
		init_streak = 0
		current_len = 0
		current_init = len(rwd) // 10
		for i in range(current_init, len(rwd)):
			if rwd[i] >= threshold:
				current_len += 1
			if current_len > max_len:
				max_len = current_len
				init_streak = current_init
			if rwd[i] < threshold:
				current_init = i
				current_len = 0
		if max_len < streak_size:
			streaks.append([len(rwd), max_len])
		else:
			streaks.append([init_streak, max_len])
	return np.array(streaks)
def test(x, y):
	# stat, p = stats.mannwhitneyu(x, y)
	stat, p = stats.ttest_ind(x, y, equal_var=False)
	print(f"t={stat}, p={p}")
	alpha = 0.05
	if p > alpha:
		print('Same distribution (fail to reject H0)')
	else:
		print('Different distribution (reject H0)')
	effect = cohend(x, y)
	alpha = 0.05
	analysis = TTestIndPower()
	power = analysis.solve_power(effect, power=None, nobs1=len(x), ratio=1.0, alpha=alpha)
	# samples = analysis.solve_power(effect, power=0.8, nobs1=None, ratio=1.0, alpha=alpha)
	dof = welch_ttest(np.array(x), np.array(y))
	print(f"effect: {effect:.2f}")
	print(f"power: {power:.2f}")
	# print(f"samples: {samples:.2f}")
	print(f"degrees of freedom: {dof:.2f}")

def process_dir(directory):
	"""
	docstring
	"""
	files_list = sorted([f for f in listdir(input_directory) if isfile(join(input_directory, f))])
	for filepath in files_list:
		name = filepath[:-4].replace(".", "")
		splited_name = name.strip().split("_")
		delta = float(splited_name[-1]) / (float(splited_name[11]) * float(splited_name[7]))
		rewards = read_mat_from_file(join(input_directory, filepath))
		rewards_vanilla_q_learning = np.stack(rewards[0], axis=0)
		rewards_causal_q_learning = np.stack(rewards[1], axis=0)
		solving_threshold = avg_rwd_last_t_episodes(rewards_vanilla_q_learning)
		for i in [10]:
			streaks_q = max_streaks(rewards_vanilla_q_learning, solving_threshold, i)
			streaks_cm = max_streaks(rewards_causal_q_learning, solving_threshold, i)
			print(streaks_q[:,0], streaks_cm[:, 0])
			test(streaks_q[:,0], streaks_cm[:, 0])

if __name__ == "__main__":
	input_directory = sys.argv[1]
	# output_file_name = sys.argv[2]
	# experiment_name = sys.argv[3]
	# base_dir_plots = sys.argv[4]
	# mod = int(sys.argv[5])
	# n_eval_episodes = int(sys.argv[6])
	# fileout = open(output_file_name, "w")
	# alpha = 0.05
	# html_text = "<h1>{}</h1>".format(experiment_name)
	# fileout.close()
	process_dir(input_directory)

# onlyfiles = sorted([f for f in listdir(input_directory) if isfile(join(input_directory, f))])
# labels = ["Q-learning", "Q-learning + estructura completa", \
#             "Q-learning + estructura parcial", "Q-learning + estructura incorrecta"]
# for file_path in onlyfiles:
# 	name = file_path[:-4].replace(".", "")
# 	splited_name = name.strip().split("_")
# 	delta = float(splited_name[-1]) / (float(splited_name[11]) * float(splited_name[7]))
# 	html_text += "<h2>{} delta {:.2f}</h2>".format(" ".join(name.strip().split("_")), delta)
# 	rewards = read_mat_from_file(join(input_directory, file_path))
# 	# primero grafico
# 	mod_mat = transform_to_modulated_matrix(rewards, mod)
# 	mean_vectors, std_dev_vectors = compute_mean_and_std_dev(mod_mat)
# 	x_axis = mod * (np.arange(len(mean_vectors[0])))
# 	plot_path = join(base_dir_plots, name)
# 	plot_rewards(x_axis, mean_vectors, std_dev_vectors, labels, plot_path)
# 	# html_text += "<iframe src='{}.pdf'' width='50%' height='500px'></iframe>".format(plot_path)
# 	# html_text += '<object data="{}.pdf" type="application/pdf" width="700px" height="700px"><embed src="{}.pdf"><p>This browser does not support PDFs. Please download the PDF to view it: <a href="{}.pdf">Download PDF</a>.</p></embed></object>'.format(plot_path, plot_path, plot_path)
# 	mean_vectors, std_dev_vectors = compute_mean_and_std_dev(rewards)
# 	mean_eval, pvalues = compute_stat_test(mean_vectors, n_eval_episodes)

# 	array_to_table_means = []
# 	array_to_table_pvalues = []
# 	best_reward_i = np.argmax([_[0] for _ in mean_eval])
# 	i = 0
# 	for label, value in zip(labels, mean_eval):
# 		string = "{:.4f} \\pm {:.4f}".format(value[0], value[1])
# 		if pvalues[i] >= alpha and i != 0:
# 			string += "\\dagger"
# 		if i == best_reward_i:
# 			string = "\\mathbf{" + string + "}"
# 		array_to_table_means.append((label, "$" + string + "$"))
# 		i += 1
# 	for label, value in zip(labels, pvalues):
# 		is_rejected = "rejected" if value < alpha else "fail to reject"
# 		array_to_table_pvalues.append((label, "${:.4f}$".format(value), is_rejected))
# 	header = ["Algoritmo", "Recompensa promedio E = {}".format(n_eval_episodes)]
# 	html_text += array_to_html_table(header, array_to_table_means)
# 	header = ["Algoritmo", "pvalue vs Q-learning"]
# 	html_text += array_to_html_table(header, array_to_table_pvalues)
# fileout.writelines(html_text)
# fileout.close()
#leer archivos de carpeta
