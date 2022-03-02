Pedagogical teacher

> Change trajectories with goal prediction - ok.
> Adjust reward in HER transitions - ok.
> Test - ok.

>>> Works, checkpoint on github - ok.

Learner with teacher with demonstration buffer

> Teacher sample goal - ok.
> Teacher sample demo - ok.
> Learner predict goal from demo - ok.
> Learner trains with predicted goal in normal GANGSTR way (remove pedagogical stuff from train_teacher.py > self.pedagogical == False) - ok.
> Learner gets reward for correct goal prediction - ok.
> bail de no noise a vérifier - ok.

>>> Does not work. Moving on to other approach.

Learning with teacher with demonstration buffer and behaviour cloning.

> Add BC training in train_learner.py - ok.
> Create train_bc() method for BC training based on policy.train() script in rl_agent.py - ok.
> Implement BC loss - ok.
> Test with BC only on 4/5 goals - ok.
> Test with BC + RL training on 4/5 goals - ok.
> Test with BC + RL training on all goals and check that naive/pedagogical policy is imitated - ok.
> Test with BC + RL training + goal prediction on all goals and get predictability + reachability results - ok.

>>> Works for speeding up training and learn with demos in buffer but does not help in predictability/replicating the teacher's policy.

Learning with teacher with demonstration buffer and SQIL (demo reward = 1, experience = 0, 50%/50%).

> Modify reward computing in her.py - ok.
> Test as is with 1000 pedagogical demos and 1000 naive demos - ok but does not achieve the end of training.
> Test with 10k biased init pedagogical demos and 10k biased init naive demos
> Test with annealing of experience percent in buffer.

>>> 

Learning with teacher with demonstration buffer and SIL regularization.

> Implement return computation in buffer as a key in episode dictionary in her.py and unit test
> Modify reward computing in her.py (put return instead of reward) and unit test
> Verify computation of value and loss with Ahmed
> Implement prioritized replay
> Test with pedagogical demos and naive demos


