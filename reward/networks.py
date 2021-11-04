import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class RewardNetworkFlat(nn.Module):
    def __init__(self, env_params):
        super(RewardNetworkFlat, self).__init__()

        self.linear1 = nn.Linear(env_params['obs'] +  env_params['goal'], 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 4)

        self.apply(weights_init_)

    def forward(self, obs, g):
        x = torch.cat([obs, g], 1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x