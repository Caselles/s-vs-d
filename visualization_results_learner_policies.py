import matplotlib.pyplot as plt
import numpy as np

fontsize = 25

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


	'''for g in range(3):

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
			plt.close()'''

	for g in range(3):

		names = ['Purple ball', 'Orange ball', 'Pink ball']
		values = policies[str(g)]
		colors = ['purple', 'orange', 'pink']

		fig, (ax0, axR, ax1, ax2, ax3) = plt.subplots(1, 5, figsize=(40,8))
		axR.get_xaxis().set_visible(False)
		axR.get_yaxis().set_visible(False)
		axR.grid(False)
		axR.axis('off')

		axes = {-1: ax0, 0:ax1, 1:ax2, 2:ax3}
		fig.suptitle('Policy for goal '+str(g), fontsize=fontsize*1.5)

		axes[-1].set_xlabel('Action', fontsize=fontsize)
		axes[-1].set_ylabel('Probability', fontsize=fontsize)
		axes[-1].set_title('Policy for goal '+str(g)+ ' - first action', fontsize=fontsize)

		axes[-1].bar(names, values, color=colors)

		for s in range(3):

			names = ['Purple ball', 'Orange ball', 'Pink ball']
			values = policies[str(g)+'_'+str(s)]
			colors = ['purple', 'orange', 'pink']

			#axes[s].figure(figsize=(9, 9))

			axes[s].set_xlabel('Action', fontsize=fontsize)
			axes[s].set_ylabel('Probability', fontsize=fontsize)
			axes[s].set_title('Policy for goal '+str(g)+' knowing that s1='+str(s), fontsize=fontsize)

			axes[s].bar(names, values, color=colors)


		plt.savefig(plot_path + 'policies_goal_'+str(g)+'.png')
		plt.close()


