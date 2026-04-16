# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .turtlebot3_env import TurtleBot3Env, TurtleBot3EnvCfg


##
# Register Gym environments.
##

gym.register(
    id="Isaac-TurtleBot3-Direct-MVN-v0",
    entry_point="omni.isaac.lab_tasks.direct.turtlebot3_maneuver:TurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TurtleBot3EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TurtleBot3PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_sac_ddpg_cfg.yaml",
    },
)
