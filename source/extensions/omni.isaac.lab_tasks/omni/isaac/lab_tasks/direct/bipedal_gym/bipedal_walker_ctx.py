import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from rl_games.common import env_configurations
from omni.isaac.lab.utils import configclass

@configclass
class ContexWrapperCfg:
    context_dim: int = 2
    resample_every: int = 50
    low: float = -1.0
    high: float = 1.0
    device: str = "cuda:0"
    env = gym.make("BipedalWalker-v3")

class ContextWrapper(gym.Wrapper):
    def __init__(self, cfg: ContexWrapperCfg = ContexWrapperCfg()):
        super().__init__(cfg.env)
        self.context_dim = cfg.context_dim
        self.resample_every = cfg.resample_every
        self.low, self.high = cfg.low, cfg.high
        self.device = cfg.device

        # Current context vector
        self.steps = 0
        self.context = torch.zeros(self.context_dim, device=self.device)

        # Extend observation space to reflect context addition
        orig_obs_space = cfg.env.observation_space
        assert isinstance(orig_obs_space, spaces.Box), "Only Box obs supported"
        low_obs = torch.cat([
            torch.tensor(orig_obs_space.low, dtype=torch.float32),
            torch.full((self.context_dim,), self.low)
        ])
        high_obs = torch.cat([
            torch.tensor(orig_obs_space.high, dtype=torch.float32),
            torch.full((self.context_dim,), self.high)
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
        

class BipedalWalkerCtxEnv(gym.Env):
    def __init__(self, cfg: ContexWrapperCfg = ContexWrapperCfg()):
        self.env = ContextWrapper(cfg)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.num_envs = 1  # Single environment instance

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()


def make_bipedal_ctx_env(cfg):
    env = gym.make("BipedalWalker-v3")
    env = ContextWrapper(cfg)
    return env
