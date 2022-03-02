import sys 

sys.path.append('../')

import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
import numpy as np
from utils import generate_goals_demonstrator
from rollout import RolloutWorker
import json
from types import SimpleNamespace
from goal_sampler import GoalSampler
import random
from mpi4py import MPI
import pickle
from copy import deepcopy
from arguments import get_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Teacher:
    def __init__(self, args):

        self.policy = None
        self.demo_dataset = None
        self.nb_available_demos = 1000
        self.all_goals = generate_goals_demonstrator()
        if args.teacher_mode == 'naive':
            self.path_demos = '../../gangstr_predicates_instructions/demos_datasets/naive_teacher_1000/'
        if args.teacher_mode == 'pedagogical':
            self.path_demos = '../../gangstr_predicates_instructions/demos_datasets/pedagogical_teacher_1000/'

        if args.teacher_mode == 'naive_no_noise':
            self.path_demos = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/demos_datasets/naive_teacher_no_noise/'
        if args.teacher_mode == 'pedagogical_no_noise':
            self.path_demos = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/demos_datasets/pedagogical_teacher_no_noise/'


    def get_demo_for_goals(self, goals, saved=False):

        demos = []

        for goal in goals:

            if saved:

                goal_ind = self.all_goals.index(goal.tolist())

                #print(goal_ind, 'goal_ind')

                ind = np.random.randint(self.nb_available_demos)

                with open(self.path_demos + 'goal_' + str(goal_ind) + '/demo_' + str(ind) + '.pkl', 'rb') as f:
                    demo = pickle.load(f)

                # check if we get demo for the right goal
                assert (goal == demo['g'][-1]).all()

                demos.append(demo)

            else:
                raise NotImplementedError

        return demos

