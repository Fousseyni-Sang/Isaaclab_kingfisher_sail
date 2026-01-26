# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""


from omni.isaac.lab.app import AppLauncher


import gymnasium as gym
import math
import os
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.assets import retrieve_file_path
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from omni.isaac.lab_tasks.direct.kingfisher_sail_low_level.kingfisher_sail_env import KingfisherSailEnv, KingfisherSailEnvCfg

from gymnasium import spaces
import torch
import numpy as np
class DummyLowLevelEnv:
    def __init__(self, act_dim:int=3, obs_dim:int=11):
        # must match the env used during training
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

    def reset(self):
        pass

    def step(self, action):
        pass

def get_low_level_agent(num_envs: int, checkpoint_path: str, task_name:str="Isaac-KingfisherSail-Direct-Low-v0",
                        device="cuda:0", act_dim:int=3, obs_dim:int=11):
    """Play with RL-Games agent."""

    env = DummyLowLevelEnv(act_dim, obs_dim)
    agent_cfg = load_cfg_from_registry(task_name, "rl_games_cfg_entry_point")

    agent_cfg = load_cfg_from_registry(task_name=task_name, entry_point_key="rl_games_cfg_entry_point")
    
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    
    resume_path = retrieve_file_path(checkpoint_path)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.device=device
    agent.restore(resume_path)
    agent.reset()
    agent.has_batch_dimension = True

    return agent

import torch.nn as nn
import torch.nn.functional as F

class LowPPOAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()

        self.running_mean_std = nn.Identity()  # placeholder for normalization

        # main actor-critic network
        self.a2c_network = nn.ModuleDict({
            "actor_mlp": nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
            ),
            "mu": nn.Linear(hidden_dim, act_dim),
            "value": nn.Linear(hidden_dim, 1),
        })

        # sigma is a learnable parameter, not a module
        self.a2c_network_sigma = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        x = self.a2c_network["actor_mlp"](x)
        mu = torch.tanh(self.a2c_network["mu"](x))
        sigma = torch.exp(self.a2c_network_sigma)
        value = self.a2c_network["value"](x)
        return mu, sigma, value

    
    def load(self, path:str):
        checkpoint = torch.load(path, weights_only=False)
        if "model" in checkpoint:
            self.load_state_dict(checkpoint["model"])
        else:
            self.load_state_dict(checkpoint)

        self.eval()
