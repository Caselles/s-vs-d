import numpy as np
from language.build_dataset import sentence_from_configuration
from utils import language_to_id
from utils import generate_goals_demonstrator

def is_success(ag, g, mask=None):
    if mask is None:
        #return (ag == g).all()
        active_predicates_goal = np.where(g == 1)
        return (ag[active_predicates_goal] == g[active_predicates_goal]).all()
    else:
        active_predicates_goal = np.where(g == 1)
        return (ag[active_predicates_goal] == g[active_predicates_goal]).all()


class RolloutWorker:
    def __init__(self, env, policy, goal_sampler, args):

        self.env = env
        self.policy = policy
        self.env_params = args.env_params
        self.biased_init = args.biased_init
        self.goal_sampler = goal_sampler
        self.args = args
        self.pedagogical = args.pedagogical_teacher
        if self.pedagogical:
            # the goals for which we want to compute a probability conditioned on the demonstrations
            self.proba_goals = np.array(generate_goals_demonstrator())
            self.proba_masks = np.zeros((len(self.proba_goals), 9))

    def generate_rollout(self, goals, masks, self_eval, true_eval, 
        biased_init=False, animated=False, language_goal=None, verbose=False, return_proba=None, illustrative_example=False):

        episodes = []
        for index_goal, i in enumerate(range(goals.shape[0])):
            if illustrative_example:
                observation = self.env.unwrapped.reset_goal_illustrative_example(goal=np.array(goals[i]), biased_init=biased_init)
            else:
                observation = self.env.unwrapped.reset_goal(goal=np.array(goals[i]), biased_init=biased_init)
            obs = observation['observation']
            ag = observation['achieved_goal']
            ag_bin = observation['achieved_goal_binary']
            g = observation['desired_goal']
            g_bin = observation['desired_goal_binary']

            '''print('\n')
            print('\n')
            print('\n')
            print('Goal achieved in the initial state: ', ag)
            print('Goal desired g_d, pursued by the teacher:', g)'''

            # in the language condition, we need to sample a language goal
            # here we sampled a configuration goal like in DECSTR, so we just use a language goal describing one of the predicates
            if self.args.algo == 'language':
                if language_goal is None:
                    language_goal_ep = sentence_from_configuration(g, eval=true_eval)
                else:
                    language_goal_ep = language_goal[i]
                lg_id = language_to_id[language_goal_ep]
            else:
                language_goal_ep = None
                lg_id = None

            ep_obs, ep_ag, ep_ag_bin, ep_g, ep_g_bin, ep_actions, ep_success, ep_rewards = [], [], [], [], [], [], [], []
            ep_lg_id = []
            ep_masks = []

            action_probas = {}

            # Start to collect samples
            for t in range(self.env_params['max_timesteps']):
                # Run policy for one step
                no_noise = self_eval or true_eval  # do not use exploration noise if running self-evaluations or offline evaluations
                # feed both the observation and mask to the policy module
                # this is the real action
                action = self.policy.act(obs.copy(), ag.copy(), g.copy(), masks[i].copy(), no_noise, language_goal=language_goal_ep)
                action_proba = self.policy.get_action_proba(action, obs.copy(), ag.copy(), g.copy(), masks[i].copy(), no_noise, language_goal=language_goal_ep)

                if isinstance(return_proba, np.ndarray):
                    proba_list = []
                    # compute the probability of this action under the other goals to see if we can predict the goal using this proba
                    for goal in return_proba:
                        action_proba = self.policy.get_action_proba(action, obs.copy(), ag.copy(), goal.copy(), masks[i].copy(), no_noise, language_goal=language_goal_ep)
                        proba_list.append(action_proba)
                    #import pdb;pdb.set_trace()
                    action_probas[t] = proba_list
                    

                # feed the actions into the environment
                if animated:
                    self.env.render()

                observation_new, r, _, _ = self.env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                ag_new_bin = observation_new['achieved_goal_binary']

                # Append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_ag_bin.append(ag_bin.copy())
                ep_g.append(g.copy())
                ep_g_bin.append(g_bin.copy())
                ep_actions.append(action.copy())
                ep_rewards.append(r)
                ep_lg_id.append(lg_id)
                ep_success.append(is_success(ag_new, g, masks[i]))
                ep_masks.append(np.array(masks[i]).copy())

                # Re-assign the observation
                obs = obs_new
                ag = ag_new
                ag_bin = ag_new_bin

            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            ep_ag_bin.append(ag_bin.copy())

            # Gather everything
            episode = dict(obs=np.array(ep_obs).copy(),
                           act=np.array(ep_actions).copy(),
                           g=np.array(ep_g).copy(),
                           ag=np.array(ep_ag).copy(),
                           success=np.array(ep_success).copy(),
                           g_binary=np.array(ep_g_bin).copy(),
                           ag_binary=np.array(ep_ag_bin).copy(),
                           rewards=np.array(ep_rewards).copy(),
                           lg_ids=np.array(ep_lg_id).copy(),
                           masks=np.array(ep_masks).copy(),
                           self_eval=self_eval)

            '''print('Goal achieved in the final state: ', ag)'''

            if self.pedagogical:

                # check if trajectory is success (otherwise, no pedagogical reward)
                active_predicates_goal = np.where(episode['g'][-1] == 1)
                trajectory_success = (episode['ag'][-1][active_predicates_goal] == episode['g'][-1][active_predicates_goal]).all()

                # compute reward with pedagogical condition
                if trajectory_success:
                    print('Success for trajectory, computing pedagogical reward.')
                    reward_pedagogical = self.compute_reward_pedagogical(episode)
                    episode['reward_pedagogical'] = reward_pedagogical
                    if reward_pedagogical:
                        print('Pedagogical Reward is POSITIVE.')
                    else:
                        print('Pedagogical Reward is NEGATIVE!')
                else:
                    reward_pedagogical = 0
                    episode['reward_pedagogical'] = reward_pedagogical


            if self.args.algo == 'language':
                episode['language_goal'] = language_goal_ep

            episodes.append(episode)

            if verbose:
                #print('Goal desired g_d, pursued by the teacher:', ep_g[-1])
                print('Goal achieved g_a at the end of demonstration:', ep_ag[-1])
                print('Are g_a and g_d equal? >', (ep_g[-1] == ep_ag[-1]).all())
                active_predicates_goal = np.where(ep_g[-1] == 1)
                print('Are g_a and g_d loosely equal? >', (ep_ag[-1][active_predicates_goal] == ep_g[-1][active_predicates_goal]).all())
                #if (ep_g[-1] == ep_ag[-1]).all() == False and (ep_ag[-1][active_predicates_goal] == ep_g[-1][active_predicates_goal]).all() == True:
                if True:
                    #print(action_probas)
                    print(index_goal, 'INDEX GOAL')
                    total_probas = np.ones(len(action_probas[0]))
                    #for i in range(self.env_params['max_timesteps']):
                    for i in range(100):
                        total_probas *= action_probas[i]
                    print(np.argmax(total_probas), 'GOAL PREDICTION ARGMAX')
                    total_probas_sampling = total_probas/np.sum(total_probas)
                    print(np.random.choice(range(len(total_probas)), p=total_probas_sampling), 'GOAL PREDICTION SAMPLING')
                    print(total_probas_sampling)
                    from scipy.stats import entropy
                    print(entropy(total_probas_sampling), 'ENTROPY')
                '''#print('Are they loosely equal? >', (ep_ag[-1][active_predicates_goal] == ep_g[-1][active_predicates_goal]).all())
                print('Reward final', ep_rewards[-1])
                print('Episode success: ', ep_success[-1])'''
                print('\n')
                print('\n')
                print('\n')


        return episodes

    def generate_proba_from_demo(self, ep, no_noise):

        probas = np.ones(len(self.proba_goals))

        probas_timesteps = {}

        for i in range(len(ep['act'])):
            probas_timestep_t = []
            # compute the probability of this action under the other goals to see if we can predict the goal using this proba
            for j,goal in enumerate(self.proba_goals):
                action_proba = self.policy.get_action_proba(ep['act'][i], ep['obs'][i], ep['ag'][i], goal, self.proba_masks[j], no_noise)
                probas_timestep_t.append(action_proba)
            probas_timesteps[i] = probas_timestep_t

        for i in range(len(ep['act'])):
            probas *= probas_timesteps[i]

        return probas


    def compute_reward_pedagogical(self, episode):

        no_noise = False

        probas = self.generate_proba_from_demo(episode, no_noise=no_noise)

        probas = probas/np.sum(probas) # normalizing to distribution
        
        goal_prediction = self.proba_goals[np.random.choice(range(len(self.proba_goals)), p=probas)]

        if (goal_prediction == episode['g'][-1]).all():
            reward_pedagogical = 2

        else:
            reward_pedagogical = 0

        return reward_pedagogical





