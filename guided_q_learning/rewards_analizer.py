import sys
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestIndPower

from utils.performance_utils import *
from utils.vis_utils import plot_rewards, plot_boxplot
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

def local_results_to_html(filename, test_table, plot_path):
	"""
	docstring
	"""
	name = filename[:-4].replace(".", "")
	splited_name = name.strip().split("_")
	text = ""
	delta = float(splited_name[-1]) / (float(splited_name[11]) * float(splited_name[7]))
	text += f"<h3>Ambiente: {splited_name[0]}</h3>"
	text += f"<h3>Porcenjate de modificación: {splited_name[1]} {splited_name[2]}</h3>"
	text += f"<h3>Tipo de estructura: {' '.join(splited_name[3:6])} </h3>"
	text += f"<h3>N: {splited_name[7]} </h3>"
	text += f"<h3>Simulaciones: {splited_name[9]} </h3>"
	text += f"<h3>Episodios: {splited_name[11]} </h3>"
	text += f"<h3>Delta: {delta} </h3>"
	header = ["Algoritmo", "M", "SD", "t", "df", "p", "Decisión", "d de Cohen"]
	text += array_to_html_table(header, test_table)
	# text += f"\n\n![plot]({plot_path})"
	text += f'<img src="{plot_path}" title="{name}">'
	return text

def printable_list(row):
	return [ f"{cell:.2f}"  if type(cell) != str and type(cell) != int else cell for cell in row]

def push_into_storage(storage, num, i, episodes):
	labels = ["Q-learning", "Q-learning + estructura completa", \
            "Q-learning + estructura parcial", "Q-learning + estructura incorrecta"]
	storage["Algoritmo"] = np.concatenate((storage["Algoritmo"], [labels[i]]), axis=None)
	storage["N"] = np.concatenate((storage["N"], [num]), axis=None)
	storage["Episodio"] = np.concatenate((storage["Episodio"], np.mean(episodes[:, 0])), axis=None)

def run_tests(results_storage, labels, threshold, rewards, streak_size, num, mod=50):
	"""
	docstring
	"""
	vanilla_q_streak = max_streaks(rewards[0], threshold, streak_size) * mod
	push_into_storage(results_storage, num, 0, vanilla_q_streak)
	table = []
	for i in range(len(rewards)):
		causal_streak = max_streaks(rewards[i], threshold, streak_size) * mod
		if i > 0:
			push_into_storage(results_storage, num, i, causal_streak)
		printable_row = printable_list([labels[i], np.mean(causal_streak[:, 0]), np.std(causal_streak[:, 0])]\
									+ test(causal_streak[:, 0], vanilla_q_streak[:, 0]))
		table.append(printable_row)
	return table

def create_storage(labels):
	results_storage = dict(one_to_one={}, many_to_one={}, one_to_many={})
	for struct in results_storage:
		results_storage[struct] = dict(N=[], Algoritmo=[], Episodio=[])
	return results_storage

def get_struct_and_num(filename):
	splited_name = filename.strip().split("_")
	struct = "_".join(splited_name[3:6])
	num = splited_name[7]
	return struct, int(num)

def plot_mat(mat, base_dir_plots, name, mod):
	labels = ["Q-learning", "Q-learning + estructura completa", \
            "Q-learning + estructura parcial", "Q-learning + estructura incorrecta"]
	mean_vectors, std_dev_vectors = compute_mean_and_std_dev(mat)
	x_axis = mod * (np.arange(len(mean_vectors[0])))
	plot_path = join(base_dir_plots, name)
	plot_rewards(x_axis, mean_vectors, std_dev_vectors, labels, plot_path, filetype="png")
	return plot_path + ".png"

def save_str_to_doc(filename, string):
	with open(filename, "w") as f:
		f.writelines(string)

def call_boxplotting(memory, struct):
	plot_path = join(base_dir_plots, f"boxplot_{struct}")
	df = pd.DataFrame.from_dict(memory[struct])
	plot_boxplot(df, "N", "Episodio", "Algoritmo", plot_path)
	return plot_path + ".png"
def process_dir(input_directory, output_file_name, plot_dir, labels, mod_tests, mod_plot, experiment_name, size=10, t=50):
	"""
	docstring
	"""
	files_list = sorted([f for f in listdir(input_directory) if isfile(join(input_directory, f))])
	memory = create_storage(labels)
	html_str = f"<h1>{experiment_name}</h1>"
	html_str += f"<h2>Tamaño racha: {size}"
	for filepath in files_list:
		name = filepath[:-4].replace(".", "")
		rewards = preprocess_np_arrays(transform_to_modulated_matrix(read_mat_from_file(join(input_directory, filepath)), mod=mod_tests))
		solving_threshold = avg_rwd_last_t_episodes(rewards[0], t)
		struct, num = get_struct_and_num(filepath)
		table = run_tests(memory[struct], labels, solving_threshold, rewards, size, num, mod_tests)
		plot_path = plot_mat(transform_to_modulated_matrix(read_mat_from_file(join(input_directory, filepath)), mod=mod_plot), plot_dir, name, mod=mod_plot)
		html_str += local_results_to_html(filepath, table, plot_path)
	html_str += "<h2>Número de episodios en alcanzar racha de recompensas</h2>"
	for struct in ["one_to_one", "one_to_many", "many_to_one"]:
		html_str += f"<h3>{struct}</h3>"
		html_str += f'<img src="{call_boxplotting(memory, struct)}" title="boxplot_{struct}">'
	save_str_to_doc(output_file_name, html_str)

if __name__ == "__main__":
	input_directory = sys.argv[1]
	output_file_name = sys.argv[2]
	experiment_name = sys.argv[3]
	base_dir_plots = sys.argv[4]
	mod_plots = int(sys.argv[5])
	mod_tests = int(sys.argv[6])
	streak_size = int(sys.argv[7])
	t = int(sys.argv[8])
	labels = ["$Q_1$", "$Q_2$", "$Q_3$", "$Q_4$"]
	process_dir(input_directory, output_file_name, base_dir_plots, labels, mod_tests, mod_plots, experiment_name, streak_size, t)