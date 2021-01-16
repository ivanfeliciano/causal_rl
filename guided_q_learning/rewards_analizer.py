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

def preprocess_np_arrays(rewards):
	return [np.stack(r, axis=0) for r in rewards]

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

def test(x, y, alpha=0.05):
	t, p = stats.ttest_ind(x, y, equal_var=False)
	df = welch_ttest(np.array(x), np.array(y))
	decision = "$h_0$ aceptada" if p >= alpha else "$h_0$ rechazada"
	effect = cohend(y, x)
	# analysis = TTestIndPower()
	# power = analysis.solve_power(effect, power=None, nobs1=len(x), ratio=1.0, alpha=alpha)
	return [t, df, p, decision, effect]

def local_results_to_html(filename, test_table):
	"""
	docstring
	"""
	name = filename[:-4].replace(".", "")
	splited_name = name.strip().split("_")
	delta = float(splited_name[-1]) / (float(splited_name[11]) * float(splited_name[7]))
	title = f"<h2>{' '.join(name.strip().split('_')[:-2])} delta {delta:.2f}</h2>"
	header = ["Algoritmo", "M", "SD", "t", "df", "p", "Decisión", "d de Cohen"]
	table = array_to_html_table(header, test_table)

def run_tests(results_storage, labels, threshold, rewards, streak_size, num):
	"""
	docstring
	"""
	vanilla_q_streak = max_streaks(rewards[0], threshold, streak_size) * 50
	results_storage[labels[0]][num] = np.concatenate((results_storage[labels[0]][num], vanilla_q_streak[:, 0]), axis=None)
	table = []
	print()
	for i in range(len(rewards)):
		causal_streak = max_streaks(rewards[i], threshold, streak_size) * 50
		results_storage[labels[i]][num] = np.concatenate((results_storage[labels[i]][num], causal_streak[:, 0]), axis=None)
		table.append([labels[i], np.mean(causal_streak[:, 0]), np.std(causal_streak[:, 0])]\
									+ test(causal_streak[:, 0], vanilla_q_streak[:, 0]))
		print([labels[i], np.mean(causal_streak[:, 0]), np.std(causal_streak[:, 0])]\
									+ test(causal_streak[:, 0], vanilla_q_streak[:, 0]))
	return table

def create_storage(labels):
	results_storage = dict(one_to_one={}, many_to_one={}, one_to_many={})
	for struct in results_storage:
		results_storage[struct] = {l : {5 : [], 7 : [], 9 : []} for l in labels}
	return results_storage

def get_struct_and_num(filename):
	splited_name = filename.strip().split("_")
	struct = "_".join(splited_name[3:6])
	num = splited_name[7]
	return struct, int(num)
def process_dir(directory, labels, size=10):
	"""
	docstring
	"""
	files_list = sorted([f for f in listdir(input_directory) if isfile(join(input_directory, f))])
	memory = create_storage(labels)
	for filepath in files_list:
		rewards = preprocess_np_arrays(transform_to_modulated_matrix(read_mat_from_file(join(input_directory, filepath)), mod=50))
		solving_threshold = avg_rwd_last_t_episodes(rewards[0], t=50)
		struct, num = get_struct_and_num(filepath)
		table = run_tests(memory[struct], labels, solving_threshold, rewards, size, num)
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
	labels = ["$Q_1$", "$Q_2$", "$Q_3$", "$Q_4$"]
	process_dir(input_directory, labels)

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
