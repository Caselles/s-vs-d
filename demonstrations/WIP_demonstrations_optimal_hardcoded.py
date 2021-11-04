import sys 

sys.path.append('../')

import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
import numpy as np
from utils import generate_all_goals_in_goal_space
from rollout import RolloutWorker
import json
from types import SimpleNamespace
from goal_sampler import GoalSampler
import random
from mpi4py import MPI
import pickle
from copy import deepcopy
from language.utils import get_corresponding_sentences
from utils_demo import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Steps:
    def __init__(self):
        self.step0 = False
        self.step1 = False
        self.step2 = False
        self.step3 = False
        self.step_2a2 = False
        self.step_final = False


def is_above(x, y):
    """
    A function that returns whether the object x is above y
    """
    assert x.shape == y.shape

    print(x[2] - y[2], 'condition2 valueeee')

    print(np.linalg.norm(x[:2] - y[:2]) < 0.05, 'condition1')
    print(0.06 > x[2] - y[2], 'condition2')


    return np.linalg.norm(x[:2] - y[:2]) < 0.05 and 0.06 > x[2] - y[2]

def is_close(x, y):
    """
    A function that returns whether the object x is above y
    """
    assert x.shape == y.shape
    return np.linalg.norm(x[:2] - y[:2]) < 0.07

def is_close_or_above(x, y, close_or_above):

    if close_or_above:
        return is_close(x ,y)
    else:
        return is_above(x ,y)



def get_env_params(env):
    obs = env.reset()

    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params


def get_gripper_to_location_XYZ(agent_pos, target_pos_XYZ, grasped=False):

    action = [0.,0.,0.,0.]
    reached = False

    if np.sum((agent_pos-target_pos_XYZ)**2) > 0.001:

        for i in [0,1,2]:
            if agent_pos[i]<target_pos_XYZ[i]:
                action[i] = 0.2
            else:
                action[i] = -0.2

    else:
        reached = True

    if grasped:
        action[3] = -1. #keep the grasp
    
    return action, reached


def get_gripper_to_location_XY(agent_pos, target_pos_XY, grasped=False, close=False, precise=False):

    action = [0.,0.,0.,0.]
    reached = False

    if close:
        # close
        if precise:
            condition = np.sum((agent_pos[:2]-target_pos_XY[:2])**2) > 0.002
        else:
            condition = np.sum((agent_pos[:2]-target_pos_XY[:2])**2) > 0.004
    else:
        # above
        condition = np.sum((agent_pos[:2]-target_pos_XY[:2])**2) > 0.0001

    if condition:

        for i in [0,1]:
            if agent_pos[i]<target_pos_XY[i]:
                action[i] = 0.2
            else:
                action[i] = -0.2

    else:
        reached = True

    if grasped:
        action[3] = -1. #keep the grasp
    
    return action, reached


def get_gripper_to_location_Z(agent_pos, target_pos_Z, grasped=False, increment_height=False):

    action = [0.,0.,0.,0.]

    reached = False

    if increment_height:
        target_pos_Z[2] += increment_height

    print(target_pos_Z[2], agent_pos[2])

    if np.abs(agent_pos[2]-target_pos_Z[2]) > 0.01:

        if agent_pos[2]<target_pos_Z[2]:
            action[2] = 0.2
        else:
            action[2] = -0.2
    else:
        reached = True

    if grasped:
        action[3] = -1. #keep the grasp

    return action, reached

def get_gripper_out_of_the_way(agent_pos):

    reached = False
    print(agent_pos[2], 'LZKHFUUFH')

    if agent_pos[2] < 0.65:
        action = [0.,0.,0.2,0.]

    else:
        action = [0.,0.,0.,0.]
        reached = True

    return action, reached



def grasp():

    action = [0.,0.,0.1,-1.]

    return action

def release():

    action = [0.,0.,0.,0.01]

    return action

def get_close_or_above(agent_pos, cube_a, cube_b, steps, close, final=False, precise=False):

    action = [0.,0.,0.,0.0]

    if steps.step_final:
        return [0.,0.,0.,0.]

    if steps.step0 == False:
        action, reachedZ = get_gripper_out_of_the_way(agent_pos)
        if reachedZ:
            steps.step0=True

    elif steps.step1 == False:   
        action, reachedXY = get_gripper_to_location_XY(agent_pos, cube_a)
        if reachedXY:
            action, reachedZ = get_gripper_to_location_Z(agent_pos, cube_a)
            if reachedZ:
                action = grasp()
                steps.step1=True

    elif steps.step2 == False:
        action, reachedZ = get_gripper_to_location_Z(agent_pos, cube_b, grasped=True, increment_height=0.2)
        if reachedZ:
            steps.step2=True

    elif steps.step3 == False:
        action, reachedXY = get_gripper_to_location_XY(agent_pos, cube_b, grasped=True, close=close, precise=precise)
        print(reachedXY, 'reachedXY')
        if reachedXY:
            action, reachedZ = get_gripper_to_location_Z(agent_pos, cube_b, grasped=True, increment_height=0.05)
            print(reachedZ, 'reachedZ')
        print(is_close_or_above(cube_a, cube_b, close_or_above=close), 'IS ABOVE???')
        if is_close_or_above(cube_a, cube_b, close_or_above=close) and reachedXY and reachedZ:
            action = release()
            steps.step_2a2 = True
            steps.step3 = False
            steps.step2 = False
            steps.step1 = False
            steps.step0 = False
            if final:
                steps.step_final = True

    return action

def pos_close_two_cubes(cube_a, cube_b):

    pos_1 = cube_a[:3]
    pos_2 = cube_b[:3]

    t = (pos_1 + pos_2)/2

    # right format
    target_pos = [t[0], t[1], t[2], 0,0,0,0,0,0,0,0,0,0,0,0]

    return np.array(target_pos)


def get_demo_hardcoded_optimal_policy(goal, env, nb_timesteps, animated=False):

    demo = []

    steps = Steps()
    step_close = Steps()
    done = False

    observation = env.unwrapped.reset_goal(np.array(goal))
    config_initial = observation['achieved_goal'].copy()
    observation = env.unwrapped._get_obs()
    obs = observation['observation']
    ag = observation['achieved_goal']
    g = observation['desired_goal']

    for t in range(nb_timesteps):

        action = np.array([0.,0.,0.,0.]) # to define...
        action += 2*(np.random.rand(4)) - 1

        # AGENT PART

        agent_pos = obs[:3]
        agent_vel = obs[5:8]
        gripper_state = obs[3:5] #symmetric, two sides
        gripper_vel = obs[8:10] #symmetric, two sides

        # CUBES PART

        cube1 = obs[10:25]
        cube2 = obs[25:40]
        cube3 = obs[40:]

        #print('AGENT (3+2+3+2=10) (3 premiers = pos)')
        #print('CUBES (15+15+15=45) (3 premiers = pos)')

        goal = [1,1,1,1,0,0,1,0,0]

        close_predicates = goal[:3]
        above_predicates = goal[3:]

        #------------ # Hardcoded policy #

        if sum(above_predicates)>=2:
            # pyramid or stack of 3
            if sum(close_predicates)==3:
                # pyramid
                cube_close_a, cube_close_b, cube_above = pyramid(goal, cube1, cube2, cube3)
                if steps.step_2a2 == False:
                    action = get_close_or_above(agent_pos, cube_close_a, cube_close_b, steps, close=True)
                else:
                    target_pos = pos_close_two_cubes(cube_close_a, cube_close_b)
                    action = get_close_or_above(agent_pos, cube_above, target_pos, steps, close=False, final=True)
            else:
                # stack of three
                cube_above, cube_middle, cube_below = stack3(goal, cube1, cube2, cube3)
                if steps.step_2a2 == False:
                    action = get_close_or_above(agent_pos, cube_above, cube, steps, close=False)
                else:
                    action = get_close_or_above(agent_pos, cube_above_above, cube_above, steps, close=False, final=True)

        else:

            if sum(above_predicates)==1:
                # stack of 2

                if sum(close_predicates)==1:
                    # without a close one to the stack
                    cube_a, cube_b = stack_1far(goal, cube1, cube2, cube3)
                    action = get_close_or_above(agent_pos, cube_a, cube_b, steps, close=False)

                elif sum(close_predicates)==3: 
                    # with one close to the stack
                    cube_a, cube_b, cube_close = stack_1close(goal, cube1, cube2, cube3)
                    if steps.step_2a2 == False:
                        action = get_close_or_above(agent_pos, cube_b, cube_close, steps, close=True)
                    else:
                        action = get_close_or_above(agent_pos, cube_a, cube_b, steps, close=False, final=True)
                        
                    
####################### DONE BELOW

            elif sum(close_predicates)==3:
                # all three close
                # meme chose que close 2 a 2
                # calculer la position a atteindre pour que ça soit close des deux une fois qu'on en a mis deux a coté
                cube_a, cube_b, cube_c = all3close(goal, cube1, cube2, cube3)
                if steps.step_2a2 == False:
                    action = get_close_or_above(agent_pos, cube_a, cube_b, steps, close=True)
                else:
                    target_pos = pos_close_two_cubes(cube_a, cube_b)
                    action = get_close_or_above(agent_pos, cube_c, target_pos, steps, close=True, final=True, precise=True)

            elif sum(close_predicates)==2:
                # close 2 à 2
                cube_central, cube_a, cube_b = close2a2(goal, cube1, cube2, cube3) 

                if steps.step_2a2 == False:
                    action = get_close_or_above(agent_pos, cube_central, cube_a, steps, close=True)
                else:
                    action = get_close_or_above(agent_pos, cube_b, cube_central, steps, close=True, final=True)

            elif sum(close_predicates)==1:
                # 2 close, one far

                cube_a, cube_b = close2_far1(goal, cube1, cube2, cube3)

                action = get_close_or_above(agent_pos, cube_a, cube_b, steps, close=True, final=True)

            else:
                # all far apart
                print("todo")





                



        # si c_i close de c_j > est ce qu'ils sont liés sur du above? Si oui est ce que c'est une pile? Si oui on fait la pile. Sinon
        # on applique above, sinon on applique close sinon on fait r.
        # ensuite on check les above: si deja checké, on passe, sinon on applique.

        #------------ #


        # ungrasp at the beginning
        if t < 5:
            action[3] = 1

        #print("ACTION ISSSS", action)

        # if goal is reached, do nothing
        if done:
            #action = [0.,0.,0.,0.]
            print("DONE")
            print(t)

        # demo saving
        demo.append([obs, action])

        #------------ # Environment logic #
        observation_new, _, _, info = env.step(action)
        obs = observation_new['observation']
        ag = observation_new['achieved_goal']
        ag = np.array([1 if x==1 else 0 for x in ag]) 
        print(ag)  


        # check if goal is reached
        if sum(ag == goal)==9:
            #print("SUCCESS!")
            done=True

        #print(goal)
        
        if animated:
            env.render()

    return demo



if __name__ == '__main__':


    # Make the environment
    env = gym.make('FetchManipulate3Objects-v0')
    env.reset()


    goals = generate_all_goals_in_goal_space()
    print(len(goals))

    nb_trials = 10
    animated = True
    nb_timesteps = 500

    for i in range(nb_trials):

        demo = np.array(get_demo_hardcoded_optimal_policy(goals[i], env, nb_timesteps, animated=True))

        #assert demo.shape = (nb_timesteps, 2)

    print('done.')


        