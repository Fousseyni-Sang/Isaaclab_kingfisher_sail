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
parser.add_argument(
    "--experiment_name",
    type=str,
    default=None,
    help="Override agent_cfg['params']['config']['name'] used to locate logs/rl_games/<name>/. Lets a sweep "
    "launcher point each parallel eval at its own run's checkpoints without touching the registered agent yaml.",
)

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

def _resolve_output_csv_path() -> str:
    """Name/place the eval CSV after the reward config currently in effect.

    When KINGFISHER_REWARD_CFG is set (e.g. by launch_reward_sweep.py), the CSV is
    named after and saved next to that reward config, so each swept configuration
    gets its own `all_steps_swept_<run_name>.csv` inside its own run directory.
    Falls back to `all_steps_swept_default.csv` in the current directory when the
    env var isn't set (e.g. a plain manual invocation).
    """
    reward_cfg_path = os.environ.get("KINGFISHER_REWARD_CFG")
    if reward_cfg_path:
        run_dir = os.path.dirname(os.path.abspath(reward_cfg_path))
        run_name = os.path.basename(run_dir)
    else:
        run_dir = os.getcwd()
        run_name = "default"
    return os.path.join(run_dir, f"all_steps_swept_{run_name}.csv")


def main():
    """Play with RL-Games agent."""

    csv_path = _resolve_output_csv_path()

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    if args_cli.experiment_name is not None:
        agent_cfg["params"]["config"]["name"] = args_cli.experiment_name

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    acord_dir = log_dir.split("/")
    acord_dir_name = ('/'.join(acord_dir[:-3]) + "/acord/" + '/'.join(acord_dir[-2:]))

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

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
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    # reset environment
    NUM_ANGLES = env.unwrapped.paper_wind_angle.shape[1]  # len(paper_wind_angle) in the env
    NUM_SPEEDS = env.unwrapped.paper_wind_speed.shape[1]  # len(paper_wind_speed) in the env
    #print(f"[CALCUL] num_angles: {NUM_ANGLES}:{env.unwrapped.paper_wind_angle}, num_speeds: {NUM_SPEEDS}:{env.unwrapped.paper_wind_speed}")
    EPISODES_PER_ENV = NUM_ANGLES * NUM_SPEEDS  # 45 -- one full local sweep

    env.unwrapped.is_Training = False
    env.unwrapped.is_paper_eval = True
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()

    num_envs = env.unwrapped.num_envs 
    
    print(f"\n================ Vectorized Evaluation: {num_envs} envs =================\n") 

    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    # attempt to have complete control over environment stepping. However, this removes other
    # operations such as masking that is used for multi-agent learning by RL-Games.
    # Per-env bookkeeping.
    episodes_done_per_env = np.zeros(num_envs, dtype=int)
    per_env_buffers = [[] for _ in range(num_envs)]   # current in-progress episode, per env
    all_rows = []                                      # finished rows, flat list of dicts
    prev_dones = np.zeros(num_envs, dtype=bool)         # was this env done last iteration
    
    print(f"[INFO] Target: {EPISODES_PER_ENV} episodes/env x {num_envs} envs "
        f"= {EPISODES_PER_ENV * num_envs} total episodes")
    try:
        while not np.all(episodes_done_per_env >= EPISODES_PER_ENV):
            with torch.inference_mode():
                obs_t = agent.obs_to_torch(obs)
                actions = agent.get_action(obs_t, is_deterministic=agent.is_deterministic)
                obs, rew, dones, extras = env.step(actions)
                if isinstance(obs, dict):
                    obs = obs["obs"]
        
                info = extras.get("info", {})
                dones_np = dones.detach().cpu().numpy().astype(bool)

                
                for env_id in range(num_envs):
                    # Skip envs that have already completed their full local sweep --
                    # they'll keep auto-resetting/stepping under the hood but we no
                    # longer care about their data.
                    if episodes_done_per_env[env_id] >= EPISODES_PER_ENV:
                        continue

                    robot_pos_w = info.get("robot_pos_w", torch.zeros((num_envs, 2)))[..., :2]
                    aero_force = info.get("info", {}).get("aero_force", torch.zeros((num_envs, 1, 6)))
                    aoa = extras.get("info", {}).get("aoa", torch.zeros((num_envs, )))
                    app_wind_angle = extras.get("info", {}).get("app_wind_angle", torch.zeros((num_envs, ))) # (N,) 
                    true_wind_angle_b = extras.get("info", {}).get("true_wind_angle_b", torch.zeros((num_envs, )))

                    row = dict(
                        env_id=env_id,
                        time_step=len(per_env_buffers[env_id]),
                        energy=info.get("energy", torch.zeros(num_envs))[env_id].item(),
                        distance=info.get("distance", torch.zeros(num_envs))[env_id].item(),
                        energy_context=info.get("energy_context", torch.zeros(num_envs))[env_id].item(),
                        true_wind_speed=info.get("true_wind_speed", torch.zeros(num_envs))[env_id].item(),
                        true_wind_angle_w=info.get("true_wind_angle_w", torch.zeros(num_envs))[env_id].item(),
                        rb_pos_x = robot_pos_w[env_id, 0].item(), 
                        rb_pos_y = robot_pos_w[env_id, 1].item(), 
                        sail_angle = info.get("sail_angle", torch.zeros((num_envs,)))[env_id].item(),
                        aero_force = aero_force[env_id, 0, 0].item(),
                        aoa = aoa[env_id].item(),
                        app_wind_angle = app_wind_angle[env_id].item(), 
                        true_wind_angle_b = true_wind_angle_b[env_id].item(), 

                    )
        
                    if dones_np[env_id]:
                        # This row is contaminated -- it's already the NEXT episode's
                        # post-reset state, not the finishing episode's true last
                        # step (see module docstring). Because of that, the OLD
                        # episode's last KEPT row will always be "one step before"
                        # success/timeout (e.g. distance ~0.30-0.31 when
                        # goal_reached_threshold=0.3) -- never the crossing point
                        # itself. Don't try to infer success from distance
                        # thresholding on the kept rows; read the sim's own
                        # termination reason instead, which is unambiguous.
                        goal_reached = bool(env.unwrapped.reset_terminated[env_id].item())
                        timed_out = bool(env.unwrapped.reset_time_outs[env_id].item())
        
                        finished_episode_rows = per_env_buffers[env_id]
                        local_ep_idx = episodes_done_per_env[env_id]
                        for r in finished_episode_rows:
                            r["local_episode_idx"] = local_ep_idx
                            r["goal_reached"] = goal_reached
                            r["timed_out"] = timed_out
                        all_rows.extend(finished_episode_rows)
    
                        
                        episodes_done_per_env[env_id] += 1
        
                        row["time_step"] = 0
                        per_env_buffers[env_id] = [row]

                        if episodes_done_per_env.sum()%10==0:
                            print(f"[INFO] episode finished: {episodes_done_per_env.sum()}:{episodes_done_per_env}")
                    else:
                        per_env_buffers[env_id].append(row)
        
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        
        print("[INFO] All envs completed their full local wind sweep.")
        df = pd.DataFrame(all_rows)
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved to '{csv_path}': {len(df)} rows, "
            f"{df.groupby(['env_id','local_episode_idx']).ngroups} episodes total.")
        
        print("[INFO] Evaluation complete.")

        # close the simulator
        env.close()

    except KeyboardInterrupt:
        print(f"[INTERRUPT] User pressed Crtl")
        
        print(f"[INFO] episode finished: {episodes_done_per_env.sum()}:{episodes_done_per_env}")

        df = pd.DataFrame(all_rows)
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved to '{csv_path}': {len(df)} rows, "
            f"{df.groupby(['env_id','local_episode_idx']).ngroups} episodes total.")
        
        print("[INFO] Evaluation complete.")

        # close the simulator
        env.close()

if __name__ == "__main__":

    # run the main function
    main()
    # close sim app
    simulation_app.close()
