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
			
			
class RandomPolicy():

	def __init__(self, env):
	
		self.action_proba = [1]*9 / 9
		self.env = env
		
	def update_policy(action, goal, goal_achieved):
	
		return
		
	def act(self):
	
		action = np.random.choice(self.env.action_space_high_level, p=self.action_proba)
		
		return action
						
class Policy():

	# Is a MAB with respect to the target goal

	def __init__(self, env, pedagogical):
	
		self.action_proba = np.array([1]*9) / 9
		self.env = env
		self.pedagogical = pedagogical
		
	def update_policy(self, action, goal, goal_achieved):
	
		# update rule
		action_index = self.env.action_space_high_level.index(action)
		success = self.success_condition(goal, goal_achieved)
		if success:
		#if goal == goal_achieved:
		#if goal in goal_achieved:
			self.action_proba[action_index] += 0.01
		else:
			if self.action_proba[action_index] > 0.01:
				self.action_proba[action_index] -= 0.01
		self.normalize()
		
		return
	
	def act(self):
	
		action_index = np.random.choice(len(self.env.action_space_high_level), p=self.action_proba)
		action = self.env.action_space_high_level[action_index]
		
		return action
		
	def normalize(self):
	
		#self.action_proba = np.exp(self.action_proba)
	
		sum_action_proba = np.sum(self.action_proba)
		
		self.action_proba = self.action_proba / sum_action_proba
		
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

	def __init__(self, env, pedagogical):
	
		self.pedagogical = pedagogical
		self.env = env
		self.discovered_goals = [[0]]
		
		self.policy_0 = Policy(self.env, pedagogical)
		self.policy_1 = Policy(self.env, pedagogical)
		self.policy_1_2 = Policy(self.env, pedagogical)

		self.policies = {0: self.policy_0, 1:self.policy_1, 2:self.policy_1_2}
		
		
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
	
	def update_policy(self, action, goal, goal_achieved):
	
		self.policies[goal[-1]].update_policy(action, goal, goal_achieved)
		
		return
		
	
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
			print("Action proba:", formatted_list)
				
