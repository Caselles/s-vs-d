import matplotlib.pyplot as plt
import numpy as np

fontsize = 30

for p in ['optimal_literal/', 'optimal_pragmatic/', 'pedagogical_literal/', 'pedagogical_pragmatic/']:

	save_path = 'results/learner/' + p

	plot_path = 'plots/learner/' + p

	from pathlib import Path
	Path(plot_path).mkdir(parents=True, exist_ok=True)

	policies = {}

	for g in range(3):

		policies[str(g)] = np.load(save_path + 'goal_'+str(g)+'_action_proba.npy')

		for s in range(3):

			policies[str(g)+'_'+str(s)] = np.load(save_path + 'goal_'+str(g)+'_action_proba_knowing_'+str(s)+'.npy')


	names = ['Purple ball', 'Orange ball', 'Pink ball']
	values = [1/3, 1/3, 1/3]
	colors = ['purple', 'orange', 'pink']

	plt.figure(figsize=(9, 9))

	plt.xlabel('Action', fontsize=fontsize)
	plt.ylabel('Probability', fontsize=fontsize)
	plt.ylim([0,1])
	plt.title('Uniform policy', fontsize=fontsize)

	plt.bar(names, values, color=colors)

	plt.savefig(plot_path + 'uniform_policy.png')
	plt.close()


	for g in range(3):

		names = ['Purple ball', 'Orange ball', 'Pink ball']
		values = policies[str(g)]
		colors = ['purple', 'orange', 'pink']

		plt.figure(figsize=(9, 9))

		plt.xlabel('Action', fontsize=fontsize)
		plt.ylabel('Probability', fontsize=fontsize)
		plt.ylim([0,1])
		plt.title('Policy for goal '+str(g), fontsize=fontsize)

		plt.bar(names, values, color=colors)

		plt.savefig(plot_path + 'action_proba_goal_'+str(g)+'.png')
		plt.close()

		for s in range(3):

			names = ['Purple ball', 'Orange ball', 'Pink ball']
			values = policies[str(g)+'_'+str(s)]
			colors = ['purple', 'orange', 'pink']

			plt.figure(figsize=(9, 9))

			plt.xlabel('Action', fontsize=fontsize)
			plt.ylabel('Probability', fontsize=fontsize)
			plt.ylim([0,1])
			plt.title('Policy for goal '+str(g)+' knowing that s1='+str(s), fontsize=fontsize)

			plt.bar(names, values, color=colors)

			plt.savefig(plot_path + 'action_proba_goal_'+str(g)+'_knowings1equals_'+str(s)+'.png')
			plt.close()


