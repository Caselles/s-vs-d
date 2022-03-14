import numpy as np


class BallsEnv():

	def __init__(self):
	
		self.action_space = [0,1,2]
		self.action_space_high_level = [[0,0],[0,1],[0,2],[1,0],[2,2],
		[2,0],[1,1],[2,1],[1,2]]

		
	def step(self, action):
	
		if action == [1,1]:
		
			return 1, [1]
			
		elif action == [2,1]:
		
			return 1, [1]
			
		elif action == [1,2]:
		
			return 1, [1,2]
			
		else:
		
			return 0, [0]
			
		
class PolicyTeacher():

	# Is a MAB with respect to the target goal

	def __init__(self, env, pedagogical):
	
		self.action_proba = np.array([1]*3) / 3
		self.action_proba_knowing_s1 = {0:np.array([1]*3) / 3,
		1:np.array([1]*3) / 3,
		2:np.array([1]*3) / 3}
		
		self.env = env
		self.pedagogical = pedagogical
		
	def update_policy(self, action, goal, goal_achieved, belief_update=False):
	
		# update rule
		if belief_update:
			self.action_proba[action[0]] += 0.01
			
		success = self.success_condition(goal, goal_achieved)
		if success:
			self.action_proba[action[0]] += 0.01
			self.action_proba_knowing_s1[action[0]][action[1]] += 0.01
		else:
			if self.action_proba[action[0]] > 0.01:
				self.action_proba[action[0]] -= 0.01
			if self.action_proba_knowing_s1[action[0]][action[1]] > 0.01:
				self.action_proba_knowing_s1[action[0]][action[1]] -= 0.01
			
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
		if self.pedagogical:
			if goal != goal_achieved:			
				success = False
		else:
			for g in goal:
				if g not in goal_achieved:
					success = False
	
		return success
	
				
			
class Teacher():

	def __init__(self, env, pedagogical, load_ckpt=False):
	
		self.pedagogical = pedagogical
		self.env = env
		self.discovered_goals = [[0]]
		
		self.policy_0 = PolicyTeacher(self.env, pedagogical)
		self.policy_1 = PolicyTeacher(self.env, pedagogical)
		self.policy_1_2 = PolicyTeacher(self.env, pedagogical)

		self.policies = {0: self.policy_0, 1:self.policy_1, 2:self.policy_1_2}

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
		
	def sample_goal(self, learner):

		if learner:

			if self.pedagogical:
				sampled_goal = [np.random.choice(3, p=[0.04,0.48,0.48])]
			else:
				sampled_goal = [np.random.choice(3)]
		
		else:
		
			sampled_goal_index = np.random.choice(len(self.discovered_goals))
			
			sampled_goal = self.discovered_goals[sampled_goal_index]
	
		return sampled_goal
		
	def update_discovered_goals(self, goal_achieved):
	
		if goal_achieved not in self.discovered_goals:
		
			self.discovered_goals.append(goal_achieved)
		
		return
	
	def update_policy(self, action, goal, goal_achieved):
	
		if self.pedagogical:
			belief_update = self.compute_belief_update(action, goal)
		else:
			belief_update = False
	
		self.policies[goal[-1]].update_policy(action, goal, goal_achieved, belief_update)
		
		return
		
	def compute_belief_update(self, action, goal):

		# does the current trajectory allows to better predict the desired goal?
		
		# p(g/a) propto p(a/g) (can p(a/g) be discriminative? we enforce it if possible)
		proba_goal_knowing_action = []
		
		for i in range(3):
			proba_goal_knowing_action.append(self.policies[i].action_proba[action[0]])
		
		if np.argmax(proba_goal_knowing_action) == goal[-1]:
			belief_update = True
		else:
			belief_update = False
	
		return belief_update

	def get_demonstration(self, goal):

		demo = self.policies[goal[-1]].act()

		return demo
		
	
goal_mapping = {0:'Rien', 1:'Goal ambigu', 2:'Goal max'}

if __name__ == '__main__':

	verbose = 0

	env = BallsEnv()
	
	for pedagogical in [False, True]:
	
		teacher = Teacher(env, pedagogical=pedagogical)
		
		total_reward = []
		
		for i in range(1000):
		
			goal_desired = teacher.sample_goal()
			
			action = teacher.act(goal_desired)
			
			if verbose:
				print('TARGET', goal_mapping[goal_desired[-1]])
				
				formatted_list = []
				for item in teacher.policies[goal_desired[-1]].action_proba:
				    formatted_list.append("%.2f"%item)
				print("Action proba:", formatted_list)
			
			reward, goal_achieved = env.step(action)
			
			if verbose:
				print('ACHIEVED', goal_mapping[goal_achieved[-1]])
			
			success = teacher.policies[goal_desired[-1]].success_condition(goal_desired, goal_achieved)
			
			if verbose:
				print('SUCCESS', success)
			
			teacher.update_discovered_goals(goal_achieved)
			
			teacher.update_policy(action, goal_desired, goal_achieved)
			
			total_reward.append((goal_desired == goal_achieved))
			
			if verbose:
				print(np.mean(total_reward), 'Mean total reward')
			
		print('Finished training for teacher with pedagogic switch on', pedagogical, '. Summary of learned policy...')
		
		for g in teacher.discovered_goals:
		
			print('GOAL', goal_mapping[g[-1]])
			
			formatted_list = []
			for item in teacher.policies[g[-1]].action_proba:
			    formatted_list.append("%.2f"%item)
			print("Action proba first state:", formatted_list)
			
			for i in range(3):
				formatted_list = []
				for item in teacher.policies[g[-1]].action_proba_knowing_s1[i]:
				    formatted_list.append("%.2f"%item)
				print("Action proba second state:", formatted_list)


		# saving results

		from pathlib import Path
		if pedagogical:
			save_path = 'results/teacher/pedagogical/'
		else:
			save_path = 'results/teacher/optimal/'
		Path(save_path).mkdir(parents=True, exist_ok=True)

		for g in range(3):

			np.save(save_path + 'goal_'+str(g)+'_action_proba', teacher.policies[g].action_proba)

			for s in range(3):

				np.save(save_path + 'goal_'+str(g)+'_action_proba_knowing_'+str(s), teacher.policies[g].action_proba_knowing_s1[s])
				
			
				
