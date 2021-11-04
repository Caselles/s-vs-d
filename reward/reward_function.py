import torch
from torch import nn
import numpy as np
from mpi_utils.mpi_utils import sync_networks
from reward.reward_buffer import MultiRewardBuffer
from reward.networks import RewardNetworkFlat
from rl_modules.networks import QNetworkFlat, GaussianPolicyFlat
from mpi_utils.normalizer import normalizer
from her_modules.WIP_her import her_sampler
from updates import update_flat, update_deepsets
from utils import id_to_language


"""
Reward function module (MPI-version)
"""

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class RewardFunction:
    def __init__(self, args):

        self.args = args
        self.env_params = args.env_params

        self.total_iter = 0

        self.reward_loss_mean = 0


        # create the network
        self.architecture = self.args.architecture_reward_func

        if self.architecture == 'flat':
            self.cross_entropy_loss = nn.CrossEntropyLoss()
            self.model = RewardNetworkFlat(args.env_params)
            self.reward_func_optim = torch.optim.Adam(list(self.model.parameters()),
                                                 lr=self.args.lr_reward_func)
        else:
            raise NotImplementedError


        # if use GPU
        if self.args.cuda:
            #self.actor_network.cuda()
            pass


        # create the buffer
        self.buffer = MultiRewardBuffer(env_params=self.env_params,
                                  buffer_size=self.args.buffer_size)

    
    def store(self, batch):
        self.buffer.store_batch(batch=batch)

    def sample(self, batch_size):
        batch = self.buffer.sample(batch_size=batch_size)

        return batch

    # pre_process the inputs
    def _preproc_inputs(self, obs, ag, g):
        obs_norm = self.o_norm.normalize(obs)
        ag_norm = self.g_norm.normalize(ag)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, ag_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    def train(self):
        # train the network
        self.total_iter += 1
        self._update_network()
        if self.total_iter % 100 == 0:
            print(self.reward_loss_mean/self.total_iter)

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # update the network
    def _update_network(self):

        # sample from buffer and reformat
        data = self.buffer.sample(self.args.batch_size)

        inputs_obs = torch.tensor(list(data[:,0]), dtype=torch.float32)
        inputs_g = torch.tensor(list(data[:,1]), dtype=torch.float32)
        target_r = torch.tensor(list(data[:,2]), dtype=torch.long)

        #import pdb;pdb.set_trace()

        if self.args.cuda:
            inputs_obs_g = inputs_obs_g.cuda()
            target_r = target_r.cuda()

        # forward pass
        self.model.train() 
        pred_r = self.model(inputs_obs, inputs_g)

        # loss computing
        reward_func_loss = self.cross_entropy_loss(pred_r, target_r.reshape(self.args.batch_size))
        self.reward_loss_mean += reward_func_loss.item()

        # optimization
        self.reward_func_optim.zero_grad()
        reward_func_loss.backward()
        #sync_grads(reward_func)
        self.reward_func_optim.step()

        return reward_func_loss.item()


    def save(self, model_path, epoch):
        # Store model
        if self.args.architecture_reward_func == 'flat':
            torch.save([self.model.state_dict()],
                       model_path + '/model_reward_func_{}.pt'.format(epoch))
        else:
            raise NotImplementedError
        return
        

    def load(self, model_path, args):
        # Load model
        if args.architecture == 'flat':
            reward_func_model = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(reward_func_model)
        else:
            raise NotImplementedError
        return
