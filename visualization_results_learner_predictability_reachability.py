import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(20, 9))
fontsize = 30

scores_r_dict = {}
scores_p_dict = {}
algorithms = []



for pragmatic in [False, True]:
	for pedagogical in [False, True]:
		score_r = np.load('results/learner/' + 'pragmatic_'+str(int(pragmatic))+'_pedagogical_'+str(int(pedagogical))+'_reachability.npy')

		print(np.mean(score_r, axis=0), 
			'Mean reachability score for learner pragmatic'+ str(int(pragmatic)) + '+ teacher pedagogical' + str(int(pedagogical)))

		if pragmatic:
			word0 = 'Pragmatic learner'
		else:
			word0 = 'Literal learner'

		if pedagogical:
			word1 = 'Pedagogical teacher'
		else:
			word1 = 'Naive teacher'

		scores_r_dict[word0+word1] = score_r.reshape((150,1,20))
		algorithms.append(word0+word1)

		avg = np.mean(score_r, axis=0)
		plt.plot(range(len(avg)), avg, label=word0 + '+' + word1)
		std = np.nanstd(score_r, axis=0)
		plt.fill_between(range(len(avg)), avg - std, avg + std, alpha = 0.25)


plt.legend(loc="lower right", fontsize=fontsize)
plt.title('Reachability results', fontsize=fontsize)
plt.xlabel('timesteps (x100)', fontsize=fontsize)
plt.ylabel('Success rate on test set', fontsize=fontsize)
#plt.ylim(0,1)
plt.savefig('plots/learner/reachability.png')
plt.close()

plt.figure(figsize=(20, 9))



for pragmatic in [False, True]:
	for pedagogical in [False, True]:
		score_p = np.load('results/learner/' + 'pragmatic_'+str(int(pragmatic))+'_pedagogical_'+str(int(pedagogical))+'_predictability.npy')

		print(np.mean(score_p, axis=0), 
			'Mean predictability score for learner pragmatic'+ str(int(pragmatic)) + '+ teacher pedagogical' + str(int(pedagogical)))

		if pragmatic:
			word0 = 'Pragmatic learner'
		else:
			word0 = 'Literal learner'

		if pedagogical:
			word1 = 'Pedagogical teacher'
		else:
			word1 = 'Naive teacher'

		avg = np.mean(score_p, axis=0)
		plt.plot(range(len(avg)), avg, label=word0 + '+' + word1)
		std = np.nanstd(score_p, axis=0)
		plt.fill_between(range(len(avg)), avg - std, avg + std, alpha = 0.25)

plt.legend(loc="lower right", fontsize=fontsize)
plt.title('Predictability results', fontsize=fontsize)
plt.xlabel('timesteps (x100)', fontsize=fontsize)
plt.ylabel('Success rate on test set', fontsize=fontsize)
#plt.ylim(0,1)
plt.savefig('plots/learner/predictability.png')
plt.close()



#---------------------------------


'''

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

# Load ALE scores as a dictionary mapping algorithms to their human normalized
# score matrices across all 200 million frames, each of which is of size
# `(num_runs x num_games x 200)` where scores are recorded every million frame.
frames = np.array(range(20)) - 1
ale_frames_scores_dict = {algorithm: score[:, :, frames] for algorithm, score
                          in scores_r_dict.items()}
iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame])
                               for frame in range(scores.shape[-1])])
iqm_scores, iqm_cis = rly.get_interval_estimates(
  ale_frames_scores_dict, iqm, reps=50000)




plot_utils.plot_sample_efficiency_curve(
    frames+1, iqm_scores, iqm_cis, algorithms=algorithms,
    xlabel=r'Number of Frames (in millions)',
    ylabel='IQM Human Normalized Score')

plt.savefig('testos.pdf')'''