from unicodedata import name

from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab_tasks  # noqa
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import torch.nn.functional as F
from rl_games.algos_torch.running_mean_std import RunningMeanStd

class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, device, num_envs):
        super().__init__(env)
        self.num_envs = num_envs
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None
        

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.returned_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.returned_episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        return observations

    def step(self, action):
        observations, rewards, truncated, terminated, infos = super().step(action)
        self.episode_returns += torch.mean(rewards, dim=0)
        dones = truncated | terminated
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= torch.logical_not(dones).int() #1 - dones
        self.episode_lengths *= torch.logical_not(dones).int() #1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        action_space = envs.single_action_space.shape[1]
        observation_space = envs.single_observation_space.shape[1]
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_space, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.LayerNorm(256),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(observation_space, 256)),
            nn.LayerNorm(256),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.LayerNorm(256),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_space), std=0.01),
        )
        self.running_mean_std = RunningMeanStd((observation_space,))

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_space))

    def get_value(self, x):
        x = self.running_mean_std(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.running_mean_std(x)
        action_mean = self.actor_mean(x)
        
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
    def load(self, path):
        print("MODEL KEYS:")
        for name in self.state_dict().keys():
            print(name)
        checkpoint = torch.load(path)
        print("CHECKPOINT KEYS:")
        for name in checkpoint["model_state_dict"].keys():
            print(name)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.eval()

    def save(self, path):
        torch.save({"model_state_dict": self.state_dict()}, path)


class ExtractObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs["policy"]