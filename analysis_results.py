import numpy as np
import matplotlib.pyplot as plt
from balls_env_learner import Learner, PolicyLearner
from balls_env_teacher import Teacher, BallsEnv

fontsize = 30

for i,p in enumerate(['optimal_literal/', 'optimal_pragmatic/']):

	learner_ckpt = 'results/learner/' + p

	teacher_ckpt = 'results/teacher/optimal/'

	env = BallsEnv()

	learner = Learner(env=env, pragmatic=i, load_ckpt=learner_ckpt)

	teacher = Teacher(env=env, pedagogical=False, load_ckpt=teacher_ckpt)

	sr_p = {0:[], 1:[], 2:[]}
	errors = {0:[], 1:[], 2:[]}

	for _ in range(1000):

		for g in range(3):

			demo_goal_desired = teacher.get_demonstration([g]) # here the demonstration for desired goal is either pedagogic or not

			predicted_goal_desired = learner.predict_goal_from_demo(demo_goal_desired)

			success_goal_predictability = (predicted_goal_desired[-1] == g)

			sr_p[g].append(success_goal_predictability)
			if not success_goal_predictability:
				errors[g].append(demo_goal_desired)


	for g in range(3):
		print(np.mean(sr_p[g]))

		error = [str(x) for x in errors[g]]
		from collections import Counter
		print(Counter(error))



