# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse

import numpy
import yaml

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--ros", action="store_true", default=False, help="Enable ROS2 publishing.")
parser.add_argument("--wind_direction", type=float, default=180.0, help="True wind direction in degrees.") 
parser.add_argument("--wind_speed", type=float, default=5.0, help="True wind speed.") 
parser.add_argument("--num_episode", type=int, default=10, help="Number of episodes for evaluation.") 
parser.add_argument("--ros_publish_interval", type=int, default=10, help="ROS publish interval in steps.") 
parser.add_argument("--model_id", type=str, default=None, help="low level model to run") 

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


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

# ROS2 (optional) 
import rclpy 
from omni.isaac.lab_tasks.utils.my_utils.ros2_Node import RewardWeightSubscriber, DynamicsRlAgentPublisher 
from datetime import datetime
import pandas as pd
import numpy as np  

from classical_controller import (
    ClassicalControllerState, classical_action, PolarTable, DriveForceOptimalTrim,
)

def main():
    """Play with RL-Games agent."""

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
  

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})


    # reset environment
    NUM_ANGLES = env.unwrapped.paper_wind_angle.shape[1]  # len(paper_wind_angle) in the env
    NUM_SPEEDS = env.unwrapped.paper_wind_speed.shape[1]  # len(paper_wind_speed) in the env
    #print(f"[CALCUL] num_angles: {NUM_ANGLES}:{env.unwrapped.paper_wind_angle}, num_speeds: {NUM_SPEEDS}:{env.unwrapped.paper_wind_speed}")
    EPISODES_PER_ENV = NUM_ANGLES * NUM_SPEEDS  # 45 -- one full local sweep

    num_envs = env.unwrapped.num_envs

    # The classical analogue of the RL policy's `energy_context` sweep. Chosen to
    # span the same [0.1, 1.1]-ish range for a fair visual comparison; adjust
    # freely -- more points give a smoother classical curve at the cost of
    # EPISODES_PER_ENV x len(ASSIST_LEVELS) total episodes.
    ASSIST_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    # Path to the (wind_speed, wind_angle_deg, boat_speed) CSV behind your
    # sailing_polar_max_vmg plot, if you have it. None falls back to a fixed
    # close-hauled angle (see PolarTable/FALLBACK_UPWIND_ANGLE_DEG in
    # classical_controller.py).
    POLAR_CSV_PATH = "sail_sweep_results_1_copy.csv"  # your real polar sweep data


    env.unwrapped.is_Training = False
    env.unwrapped.is_paper_eval = True
    device = env.unwrapped.device
    polar_table = PolarTable(POLAR_CSV_PATH)
    # DriveForceOptimalTrim replaces the old single fixed max-L/D angle
    # (compute_max_ld_angle_deg) with a table of apparent-wind-angle-dependent
    # optimal sail trim -- classical_action() now expects THIS object as its
    # last argument, not a bare float. This is what the traceback's
    # AttributeError was pointing at: passing a float where classical_action
    # calls drive_trim.get_optimal_aoa_deg(...).
    #drive_trim = DriveForceOptimalTrim(env.unwrapped._sail_aerodynamics, device)

    all_rows = []

    for assist_level in ASSIST_LEVELS:
        print(f"[INFO] Classical baseline sweep: assist_level={assist_level} "
            f"({EPISODES_PER_ENV} episodes/env x {num_envs} envs)")

        # Restart this assist level's wind cycle from (angle_idx=0, speed_idx=0).
        # A plain env.reset() does NOT zero these -- they only advance on their
        # own via _reset_idx's is_paper_eval branch -- so we zero them explicitly
        # to get a clean, comparable 45-episode sweep for every assist_level.
        env.unwrapped.paper_wind_angle_idx.zero_()
        env.unwrapped.paper_wind_speed_idx.zero_()

        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        info = env.unwrapped.extras.get("info", {})

        state = ClassicalControllerState(num_envs, device=device)
        state.reset(torch.arange(num_envs, device=device), info)

        episodes_done_per_env = np.zeros(num_envs, dtype=int)
        per_env_buffers = [[] for _ in range(num_envs)]

        while not np.all(episodes_done_per_env >= EPISODES_PER_ENV):
            with torch.no_grad():
                info = env.unwrapped.extras.get("info", {})
                #actions = classical_action(info, state, assist_level, polar_table, drive_trim)
                actions = classical_action(info, state, assist_level, polar_table)
                obs, rew, dones, extras = env.step(actions)
                if isinstance(obs, dict):
                    obs = obs["obs"]

                info = extras.get("info", {})
                dones_np = dones.detach().cpu().numpy().astype(bool)

                #print(f"[DEBUG] app_wind: {info['app_wind_angle']}")
                for env_id in range(num_envs):
                    if episodes_done_per_env[env_id] >= EPISODES_PER_ENV:
                        continue

                    row = dict(
                        env_id=env_id,
                        time_step=len(per_env_buffers[env_id]),
                        energy=info["energy"][env_id].item(),
                        distance=info["distance"][env_id].item(),
                        energy_context=assist_level,   # renamed for pipeline compatibility
                        true_wind_speed=info["true_wind_speed"][env_id].item(),
                        true_wind_angle_w=info["true_wind_angle_w"][env_id].item(),
                    )

                    if dones_np[env_id]:
                        # Same reset-boundary contamination as the RL sweep: this
                        # row is already the NEXT episode's post-reset state.
                        goal_reached = bool(env.unwrapped.reset_terminated[env_id].item())
                        timed_out = bool(env.unwrapped.reset_time_outs[env_id].item())

                        finished_rows = per_env_buffers[env_id]
                        local_ep_idx = episodes_done_per_env[env_id]
                        for r in finished_rows:
                            r["local_episode_idx"] = local_ep_idx
                            r["goal_reached"] = goal_reached
                            r["timed_out"] = timed_out
                        all_rows.extend(finished_rows)

                        episodes_done_per_env[env_id] += 1

                        # This env just reset internally -- reseed the
                        # controller's per-env state (new start_pos, fresh tack)
                        # and start its new episode buffer with this row as t=0.
                        state.reset(torch.tensor([env_id], device=device), info)
                        row["time_step"] = 0
                        per_env_buffers[env_id] = [row]

                        if episodes_done_per_env.sum()%10==0:
                            print(f"[INFO] episode finished: {episodes_done_per_env.sum()}:{episodes_done_per_env}")
                    else:
                        per_env_buffers[env_id].append(row)

        print(f"[INFO] assist_level={assist_level} done: all {num_envs} envs "
            f"completed {EPISODES_PER_ENV} episodes each.")

    df = pd.DataFrame(all_rows)
    df["method"] = "classical"
    df.to_csv("classical_baseline_swept.csv", index=False)
    print(f"[INFO] Saved {len(df)} rows across {len(ASSIST_LEVELS)} assist levels "
        f"x {EPISODES_PER_ENV} episodes x {num_envs} envs.")

        
        

    # close the simulator
    env.close()

if __name__ == "__main__":

    # run the main function
    main()
    # close sim app
    simulation_app.close()