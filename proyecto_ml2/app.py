# -*- coding: utf-8 -*-
import argparse

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from q_learning import QLearning, MOD_EPISODE, THRESHOLD
from q_learning_causal import QLearningCausal
from streak import compute_streak

np.random.seed(42)


EXPECTED_STREAK_SIZE = 75
def print_streak(array_of_episodes):
	avg_episodes = []
	avg_streak_size = []
	for i in array_of_episodes:
		init_ep_streak, streak_size = compute_streak(i)
		if streak_size >= EXPECTED_STREAK_SIZE:
			avg_episodes.append(init_ep_streak)
			avg_streak_size.append(streak_size)
		else:
			avg_episodes.append(i[-1])
			avg_streak_size.append(0)
	print("Initial episode streak (mean) = {}\nStreak Size (mean) = {}".format(\
			np.mean(avg_episodes), np.mean(avg_streak_size)))
	print("Initial episode streak (std) = {}\nStreak Size (std) = {}".format(\
			np.std(avg_episodes), np.std(avg_streak_size))) 
	return avg_episodes

def plot_rewards_comparison(t_rewards_q_learning, t_rewards_q_cm, stochastic):
	scale_x = len(np.mean(t_rewards_q_learning, axis=0))
	plot_x_axis = MOD_EPISODE * (np.arange(scale_x) + 1)
	mean_q_learning = np.mean(t_rewards_q_learning, axis=0)
	std_dev_q_learning = np.std(t_rewards_q_learning, axis=0)
	mean_q_cm = np.mean(t_rewards_q_cm, axis=0)
	std_dev_q_cm = np.std(t_rewards_q_cm, axis=0)
	plt.plot(plot_x_axis, mean_q_learning, label="Vanilla Q-learning", color='#CC4F1B')
	plt.fill_between(plot_x_axis, mean_q_learning - std_dev_q_learning, mean_q_learning + std_dev_q_learning,\
		alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
	plt.plot(plot_x_axis, mean_q_cm, label="Q-learning + CM", color='#1B2ACC')
	plt.fill_between(plot_x_axis, mean_q_cm - std_dev_q_cm, mean_q_learning + std_dev_q_cm,\
		alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF')
	# plt.plot(plot_x_axis, [THRESHOLD for i in range(len(plot_x_axis))], color='green', linewidth=2)
	plt.xlabel('Episodes')
	plt.ylabel('Average Reward')
	plt.legend()
	plt.title('Average Reward Comparison {} Environment'.format("Stochastic" if stochastic else "Deterministic"))
	plt.savefig("comparison{}.svg".format("Stochastic" if stochastic else "Deterministic"), format='svg', dpi=1200)
	plt.close()

def run_significance_test(q_time_to_reach, q_cm_time_to_reach, number_of_experiments, episodes, stochastic):
	print("Computing Mann-Whitney rank test.")
	vanilla_times = q_time_to_reach
	cm_times = q_cm_time_to_reach
	for i in range(number_of_experiments):
		if cm_times[i] == None:
			cm_times[i] = episodes
		if vanilla_times[i] == None:
			vanilla_times[i] = episodes
	print("Vanilla average episode to reach goal reward: {}".format(np.mean(vanilla_times)))
	print("Vanilla standard deviation episode to reach goal reward: {}".format(np.std(vanilla_times)))
	print("CM average episode to reach goal reward: {}".format(np.mean(cm_times)))
	print("CM standard deviation episode to reach goal reward: {}".format(np.std(cm_times)))
	print(stats.mannwhitneyu(cm_times, vanilla_times))
	plt.plot(np.arange(number_of_experiments), sorted(vanilla_times), label="Vanilla Q-learning")
	plt.plot(np.arange(number_of_experiments), sorted(cm_times), label="Q-learning + CM")
	plt.xlabel('Experiments')
	plt.ylabel('Time to reach the goal (episode)')
	plt.legend()
	plt.title('Goal Reached'.format("Stochastic" if stochastic else "Deterministic"))
	plt.savefig("goal_reward_comparison{}.jpg".format("Stochastic" if stochastic else "Deterministic"))     
	plt.close()


def significance_test(q_learning, q_learning_causal):
	print(stats.mannwhitneyu(q_learning, q_learning_causal))

def main():
	parser = argparse.ArgumentParser(description='Run Q-learning and Q-learning CM to solve the classic Taxi RL problem.')
	parser.add_argument("--stochastic", help="change to simple stochastic enviroments (0.7 prob of do the choosen action)",\
						action="store_true")
	parser.add_argument("--episodes", type=int, default=1000, help="# of episodes per experiment")
	parser.add_argument("--experiments", type=int, default=5, help="# of experiments")
	parser.add_argument("--stat_test", action="store_true", help="Compute the Mann-Whitney rank test to check if some algorithm reach the goal faster. Need > 20 experiments")
	parser.add_argument("-plt", "--plot", action="store_true", help="Plot reward comparison")
	parser.add_argument("-v", "--verbose", action="store_true", help="Verbosite activated")
	args = parser.parse_args()
	episodes = args.episodes
	number_of_experiments = args.experiments
	total_rewards = [[] for i in range(2)]
	time_to_reach = [[] for i in range(2)]
	for i in range(number_of_experiments):
		if args.verbose:
			print("Running experiment {}/{}".format(i +  1, number_of_experiments))
		# print("QLearning")
		rewards_q_learning, episode_t_q = QLearning(episodes=episodes).train(plot_name="QLearning{}".format("Stochastic" if args.stochastic else ""), stochastic=args.stochastic)
		# print("QLearningCausal")
		rewards_q_learning_causal, episode_t_q_causal = QLearningCausal(episodes=episodes).train(plot_name="QLearningCausal{}".format("Stochastic" if args.stochastic else ""), stochastic=args.stochastic)
		total_rewards[0].append(rewards_q_learning)
		total_rewards[1].append(rewards_q_learning_causal)
		time_to_reach[0].append(episode_t_q)
		time_to_reach[1].append(episode_t_q_causal)
	
	print("Q-learning")
	q = print_streak(time_to_reach[0])
	print("Q-learning CM")
	qcm = print_streak(time_to_reach[1])
	significance_test(q, qcm)
	if args.plot:
		plot_rewards_comparison(total_rewards[0], total_rewards[1], args.stochastic)
	
	if args.stat_test and number_of_experiments > 20:
		pass
		# run_significance_test(time_to_reach[0], time_to_reach[1], number_of_experiments, episodes, args.stochastic)
if __name__ == '__main__':
	main()

