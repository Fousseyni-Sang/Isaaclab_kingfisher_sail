import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from rl_games.common import env_configurations

class ContextWrapper(gym.Wrapper):
    def __init__(self, env, context_dim=2, resample_every=50, low=-1.0, high=1.0, device="cuda:0"):
        super().__init__(env)
        self.context_dim = context_dim
        self.resample_every = resample_every
        self.low, self.high = low, high
        self.device = device

        # Current context vector
        self.steps = 0
        self.context = torch.zeros(self.context_dim, device=self.device)

        # Extend observation space to reflect context addition
        orig_obs_space = env.observation_space
        assert isinstance(orig_obs_space, spaces.Box), "Only Box obs supported"
        low_obs = torch.cat([
            torch.tensor(orig_obs_space.low, dtype=torch.float32),
            torch.full((context_dim,), low)
        ])
        high_obs = torch.cat([
            torch.tensor(orig_obs_space.high, dtype=torch.float32),
            torch.full((context_dim,), high)
        ])
        self.observation_space = spaces.Box(
            low=low_obs.numpy(), high=high_obs.numpy(), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.steps = 0
        self._resample_context()
        obs = self._append_context(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        if self.steps % self.resample_every == 0:
            self._resample_context()
        obs = self._append_context(obs)
        return obs, reward, terminated, truncated, info

    def _resample_context(self):
        self.context = torch.empty(self.context_dim, device=self.device).uniform_(
            self.low, self.high
        )

    def _append_context(self, obs):
        if isinstance(obs, torch.Tensor):
            return torch.cat([obs.to(self.device), self.context], dim=-1)
        else:
            # Convert numpy to torch if IsaacLab env is gym-wrapped
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            return torch.cat([obs, self.context], dim=-1)