import threading
import numpy as np
from language.build_dataset import sentence_from_configuration
from utils import language_to_id
import torch

"""
the replay buffer here is basically from the openai baselines code

"""


class MultiRewardBuffer:
    def __init__(self, env_params, buffer_size):
        self.env_params = env_params
        self.size = buffer_size

        self.current_size = 0

        # create the buffer to store info
        self.buffer = {'obs': np.empty([self.size, self.env_params['obs']]),
                       'g': np.empty([self.size, self.env_params['goal']]),
                       'r': np.empty([self.size, 1])
                       }

        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_batch(self, batch):
        batch_size = len(batch)
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)

            for i, e in enumerate(batch):
                # store the informations
                self.buffer['obs'][idxs[i]] = e[0]
                self.buffer['g'][idxs[i]] = e[1]
                self.buffer['r'][idxs[i]] = e[2]


    # sample the data from the replay buffer
    def sample(self, batch_size):

        temp_buffer = {}
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][:self.current_size]

        permutation = torch.randperm(self.current_size)
        indices = permutation[:batch_size].numpy()
        #batch = np.array([[temp_buffer['obs'][ind], temp_buffer['g'][ind], to_one_hot(temp_buffer['r'][ind])] for ind in indices])
        batch = np.array([[temp_buffer['obs'][ind], temp_buffer['g'][ind], temp_buffer['r'][ind]] for ind in indices])

        return batch

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = [idx[0]]
        return idx

def to_one_hot(r):

    one_hot = np.zeros(4)

    one_hot[int(r[0])] = 1

    return one_hot
