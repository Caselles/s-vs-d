import numpy as np
import itertools

from balls_env_teacher import Teacher, BallsEnv

class PolicyLearner():

	# Is a MAB with respect to the target goal

	def __init__(self, env):
	
		self.action_proba = np.array([1]*3) / 3
		self.action_proba_knowing_s1 = {0:np.array([1]*3) / 3,
		1:np.array([1]*3) / 3,
		2:np.array([1]*3) / 3}
		
		self.env = env
		
	def update_policy(self, action, demo_goal_desired, goal_desired, predicted_goal_desired, goal_achieved):
	
			
		# goal reachability
		success = self.success_condition(predicted_goal_desired, goal_achieved)
		if success:
			self.action_proba[action[0]] += 0.01
			self.action_proba_knowing_s1[action[0]][action[1]] += 0.01
		'''else:
			if self.action_proba[action[0]] > 0.01:
				self.action_proba[action[0]] -= 0.01
			if self.action_proba_knowing_s1[action[0]][action[1]] > 0.01:
				self.action_proba_knowing_s1[action[0]][action[1]] -= 0.01'''

		# goal predictability
		if goal_desired == predicted_goal_desired:
			self.action_proba[demo_goal_desired[0]] += 0.01
			self.action_proba_knowing_s1[demo_goal_desired[0]][demo_goal_desired[1]] += 0.01
		else:
			if self.action_proba[demo_goal_desired[0]] > 0.01:
				self.action_proba[demo_goal_desired[0]] -= 0.01
			if self.action_proba_knowing_s1[demo_goal_desired[0]][demo_goal_desired[1]] > 0.01:
				self.action_proba_knowing_s1[demo_goal_desired[0]][demo_goal_desired[1]] -= 0.01


		self.normalize()
		
		return
	
	def act(self):
	
		action_1_index = np.random.choice(len(self.env.action_space), p=self.action_proba)
		action_1 = self.env.action_space[action_1_index]
		
		
		action_2_index = np.random.choice(len(self.env.action_space), p=self.action_proba_knowing_s1[action_1])
		action_2 = self.env.action_space[action_2_index]
		
		return [action_1, action_2]
		
		
	def normalize(self):
	
		#self.action_proba = np.exp(self.action_proba)
	
		sum_action_proba = np.sum(self.action_proba)
		self.action_proba = self.action_proba / sum_action_proba
		
		for ac in self.env.action_space:
			sum_action_proba_knowing_s1 = np.sum(self.action_proba_knowing_s1[ac])
			self.action_proba_knowing_s1[ac] = self.action_proba_knowing_s1[ac] / sum_action_proba_knowing_s1
		
		return
		
	def success_condition(self, goal, goal_achieved):
	
		success = True
		for g in goal:
			if g not in goal_achieved:
				success = False
	
		return success

	def update_policy_strong_sampling(self):

		# increase proba of action 0 for goal 0 because of the sampling of the teacher

		self.action_proba[0] += 0.01
		for i in range(3):
			self.action_proba_knowing_s1[i][0] += 0.01

		self.normalize()

		return


class Learner():

	def __init__(self, env, pragmatic, load_ckpt=False):
	
		self.env = env
		self.discovered_goals = [[0]]

		self.pragmatic = pragmatic
		
		self.policy_0 = PolicyLearner(self.env)
		self.policy_1 = PolicyLearner(self.env)
		self.policy_1_2 = PolicyLearner(self.env)

		self.policies = {0: self.policy_0, 1:self.policy_1, 2:self.policy_1_2}

		self.proba_sampling_action = {0: 0.9, 1: 0.05, 2:0.05}

		perm = list(itertools.permutations([0,1,2], r=2))
		probas_demo = []

		for p in perm:
			probas_demo.append(self.proba_sampling_action[p[0]] * self.proba_sampling_action[p[1]])

		self.mean_proba_demo = np.mean(probas_demo)

		self.strong_sampling = 0
		self.strong_sampling_threshold = 5

		if load_ckpt:

			for i in range(3):

				# load action proba
				self.policies[i].action_proba = np.load(load_ckpt + 'goal_' + str(i) + '_action_proba.npy')

				for j in range(3):

					# load action proba knowing s1
					self.policies[i].action_proba_knowing_s1[j] = np.load(load_ckpt + 'goal_' + str(i) + '_action_proba_knowing_' + str(j) + '.npy')
		
		
	def act(self, goal):

		action = self.policies[goal[-1]].act()
	
		return action
		
	def sample_goal(self):
	
		sampled_goal_index = np.random.choice(len(self.discovered_goals))
		
		sampled_goal = self.discovered_goals[sampled_goal_index]
	
		return sampled_goal
		
	def update_discovered_goals(self, goal_achieved):
	
		if goal_achieved not in self.discovered_goals:
		
			self.discovered_goals.append(goal_achieved)
		
		return
	
	def update_policy(self, action, demo_goal_desired, goal_desired, predicted_goal_desired, goal_achieved):
	
		self.policies[predicted_goal_desired[-1]].update_policy(action, demo_goal_desired, goal_desired, predicted_goal_desired, goal_achieved)

		return
		
	def predict_goal_from_demo(self, demo_goal_desired):

		# is the learner able to predict the desired goal from a demonstration?
		
		# p(g/a) propto p(a/g) is the prediction of the learner given its current policy
		proba_goal_knowing_action = []
		
		for i in range(3):
			proba_goal_knowing_action.append(self.policies[i].action_proba[demo_goal_desired[0]] * self.policies[i].action_proba_knowing_s1[demo_goal_desired[0]][demo_goal_desired[1]])
		
		#predicted_goal_desired = np.argmax(proba_goal_knowing_action)
		#import pdb;pdb.set_trace()
		sum_action_proba = np.sum(proba_goal_knowing_action)
		proba_goal_knowing_action = proba_goal_knowing_action / sum_action_proba
		predicted_goal_desired = np.random.choice(range(len(proba_goal_knowing_action)), p=proba_goal_knowing_action)
	
		return [predicted_goal_desired]

	def strong_sampling_detector(self, demo):

		proba_demo = self.proba_sampling_action[demo[0]] * self.proba_sampling_action[demo[1]]

		if proba_demo < self.mean_proba_demo:
			self.strong_sampling += 1
		else:
			self.strong_sampling = 0			

		return self.strong_sampling > self.strong_sampling_threshold

	def update_policy_strong_sampling(self):

		self.policies[0].update_policy_strong_sampling()

		return


def test_learner_after_training(learner, teacher):

	nb_tests_per_goal = 100

	sr_predictability = []

	sr_reachability = []

	for g in range(3):

		for t in range(nb_tests_per_goal):

			demo_goal_desired = teacher.get_demonstration([g])

			predicted_goal_desired = learner.predict_goal_from_demo(demo_goal_desired)

			action = learner.act([g])

			reward, goal_achieved = env.step(action)

			success_goal_predictability = (predicted_goal_desired[-1] == g)

			success_goal_reachability = learner.policies[0].success_condition([g], goal_achieved)

			sr_predictability.append(success_goal_predictability)

			sr_reachability.append(success_goal_reachability)

	#print(np.mean(sr_predictability), 'Mean predictability')

	#print(np.mean(sr_reachability), 'Mean reachability')


	return np.mean(sr_predictability), np.mean(sr_reachability)

goal_mapping = {0:'Rien', 1:'Goal ambigu', 2:'Goal max'}

if __name__ == '__main__':

	verbose = 0

	env = BallsEnv()

	scores_r_pedagogical = []
	scores_p_pedagogical = []
	scores_r_optimal = []
	scores_p_optimal = []

	scores = {'pragmatic_0_pedagogical_0_reachability':[],'pragmatic_0_pedagogical_1_reachability':[],'pragmatic_1_pedagogical_0_reachability':[],
	'pragmatic_1_pedagogical_1_reachability':[],
	'pragmatic_0_pedagogical_0_predictability':[],'pragmatic_0_pedagogical_1_predictability':[],'pragmatic_1_pedagogical_0_predictability':[],
	'pragmatic_1_pedagogical_1_predictability':[]}

	scores_seed = {}

	for seed in range(150):

		for pragmatic in [False, True]:

			for pedagogical in [False, True]:

				reach_curve = []
				pred_curve = []

				learner = Learner(env, pragmatic)

				if pedagogical:
					ckpt = 'results/teacher/pedagogical/'
				else:
					ckpt = 'results/teacher/optimal/'
				teacher = Teacher(env, pedagogical, load_ckpt=ckpt)

				for i in range(2000):

					goal_desired = teacher.sample_goal(learner=True) # here the goal sampling is either pedagogic or not

					demo_goal_desired = teacher.get_demonstration(goal_desired) # here the demonstration for desired goal is either pedagogic or not

					predicted_goal_desired = learner.predict_goal_from_demo(demo_goal_desired)

					action = learner.act(predicted_goal_desired)

					reward, goal_achieved = env.step(action)

					learner.update_discovered_goals(goal_achieved)
						
					learner.update_policy(action, demo_goal_desired, goal_desired, predicted_goal_desired, goal_achieved)

					if learner.pragmatic:
						if learner.strong_sampling_detector(demo_goal_desired):
							learner.update_policy_strong_sampling()

					'''if i % 1000 == 0:

						print('Testing for learner + teacher with pedagogic mode on ', pedagogical)

						test_learner_after_training(learner, teacher)'''

					if i % 100 == 0:

						print('Testing for learner + teacher with pedagogic mode on ', pedagogical, 'and pragmatic mode on ', pragmatic)

						sr_predictability, sr_reachability = test_learner_after_training(learner, teacher)

						reach_curve.append(sr_reachability)
						pred_curve.append(sr_predictability)

				for g in learner.discovered_goals:
					
					print('GOAL', goal_mapping[g[-1]])
						
					formatted_list = []
					for item in learner.policies[g[-1]].action_proba:
						formatted_list.append("%.2f"%item)
					print("Action proba first state:", formatted_list)
						
					for i in range(3):
						formatted_list = []
						for item in learner.policies[g[-1]].action_proba_knowing_s1[i]:
							formatted_list.append("%.2f"%item)
						print("Action proba second state:", formatted_list)

				if seed == 0:
					# saving policies

					from pathlib import Path
					if pedagogical:
						if pragmatic:
							save_path = 'results/learner/pedagogical_pragmatic/'
						else:
							save_path = 'results/learner/pedagogical_literal/'
					else:
						if pragmatic:
							save_path = 'results/learner/optimal_pragmatic/'
						else:
							save_path = 'results/learner/optimal_literal/'
					Path(save_path).mkdir(parents=True, exist_ok=True)

					for g in range(3):

						np.save(save_path + 'goal_'+str(g)+'_action_proba', learner.policies[g].action_proba)

						for s in range(3):

							np.save(save_path + 'goal_'+str(g)+'_action_proba_knowing_'+str(s), learner.policies[g].action_proba_knowing_s1[s])

				scores['pragmatic_'+str(int(pragmatic))+'_pedagogical_'+str(int(pedagogical))+'_reachability'].append(reach_curve)
				scores['pragmatic_'+str(int(pragmatic))+'_pedagogical_'+str(int(pedagogical))+'_predictability'].append(pred_curve)


	for pragmatic in [False, True]:
		for pedagogical in [False, True]:
			print(np.mean(scores['pragmatic_'+str(int(pragmatic))+'_pedagogical_'+str(int(pedagogical))+'_reachability'], axis=1), 
				'Mean reachability score for learner pragmatic'+ str(int(pragmatic)) + '+ teacher pedagogical' + str(int(pedagogical)))
			np.save('results/learner/' + 'pragmatic_'+str(int(pragmatic))+'_pedagogical_'+str(int(pedagogical))+'_reachability',
			 scores['pragmatic_'+str(int(pragmatic))+'_pedagogical_'+str(int(pedagogical))+'_reachability'])
			

			print(np.mean(scores['pragmatic_'+str(int(pragmatic))+'_pedagogical_'+str(int(pedagogical))+'_predictability'], axis=1), 
				'Mean predictability score for learner pragmatic'+ str(int(pragmatic)) + '+ teacher pedagogical' + str(int(pedagogical)))
			np.save('results/learner/' + 'pragmatic_'+str(int(pragmatic))+'_pedagogical_'+str(int(pedagogical))+'_predictability',
			 scores['pragmatic_'+str(int(pragmatic))+'_pedagogical_'+str(int(pedagogical))+'_predictability'])

	import pdb;pdb.set_trace()



