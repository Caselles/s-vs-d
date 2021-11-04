import numpy as np
from mpi4py import MPI
import env
import gym
import os
from arguments import get_args
from rl_modules.rl_agent import RLAgent
import random
import torch
from rollout import RolloutWorker
from temporary_lg_goal_sampler import LanguageGoalSampler
from goal_sampler import GoalSampler
from utils import init_storage, get_instruction, get_eval_goals
import time
from mpi_utils import logger
from language.build_dataset import sentence_from_configuration

# Make the environment
env_name = 'FetchManipulate{}Objects-v0'.format(3)
env = gym.make(env_name)

import pdb;pdb.set_trace()