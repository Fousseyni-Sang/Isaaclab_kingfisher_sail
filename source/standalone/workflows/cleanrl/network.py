import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# SAC Networks
class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))
        
def to_tensor_batch(batch_column):
    return torch.stack([
        item["policy"] if isinstance(item, dict) else item
        for item in batch_column
    ])

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, action_scale, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, act_dim)
        self.fc_logstd = nn.Linear(hidden_dim, act_dim)
        self.LOG_STD_MIN = -5
        self.LOG_STD_MAX = 2
        self.action_scale = action_scale

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6)
        return torch.clamp(action * self.action_scale, -1.0, 1.0), log_prob.sum(dim=-1, keepdim=True)