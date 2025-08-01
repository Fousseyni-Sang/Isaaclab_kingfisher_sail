# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from omni.isaac.lab.app import AppLauncher

# sac_isaaclab.py

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run")
parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total number of timesteps to run")
parser.add_argument("--episode_length", type=int, default=None, help="length of episode in seconds, if None, " \
"use the default from the task config")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--buffer_size", type=int, default=20_000_000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--policy_frequency", type=int, default=2, help="the frequency of training policy (delayed)")
parser.add_argument("--target_network_frequency", type=int, default=1, help="the frequency of updates for the target nerworks")
parser.add_argument("--task_dir_name", type=str, default="kingfisher_sail_direct", help="directory name for the task")
parser.add_argument("--checkpoint", type=str, default=None, help="path to the checkpoint file to load")
parser.add_argument("--alpha", type=float, default=0.2, help="Entropy regularization coefficient")
parser.add_argument("--autotune", type=bool, default=True, help="automatic tuning of the entropy coefficient")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--use_last_checkpoint", type=bool, default=False, help="Load latest checkpoint from checkpoint_dir")
parser.add_argument("--actor_hdim", type=int, default=64, help="hidden dimension for the actor network")
parser.add_argument("--critic_hdim", type=int, default=64, help="hidden dimension for the critic network")
parser.add_argument("--wind_direction", type=float, default=180, help="direction of the true wind in degree.")

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args)
#simulation_app = app_launcher.app


import os
import time
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab_tasks  # noqa
import gymnasium as gym
from buffers import ReplayBuffer
from network import SoftQNetwork, Actor


def find_latest_checkpoint_dir(base_root):
    """
    Finds the latest 'last_*' checkpoint under logs/cleanrl/sac/{task_dir_name}/*/nn/
    """
    candidate_dirs = []
    for time_dir in os.listdir(base_root):
        nn_dir = os.path.join(base_root, time_dir, "nn")
        if not os.path.isdir(nn_dir):
            continue
        for subdir in os.listdir(nn_dir):
            if subdir.startswith("last_"):
                full_path = os.path.join(nn_dir, subdir)
                candidate_dirs.append(full_path)

    if not candidate_dirs:
        raise FileNotFoundError(f"No checkpoint directories found under {base_root}/**/nn/")

    # Sort by modification time
    latest = max(candidate_dirs, key=os.path.getmtime)
    return latest

def main():
    
    # Prepare IsaacLab environment
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs, device=args.device, use_fabric=not args.disable_fabric)
    env = gym.make(args.task, cfg=env_cfg)
    obs_space = env.observation_space
    act_space = env.action_space
    #print(f"================== obs_space: {obs_space.shape}, act_space: {act_space.shape}")
    obs_dim = obs_space.shape[1]
    act_dim = act_space.shape[1]
    action_scale = 1 #torch.tensor((act_space.high - act_space.low) / 2.0, device=args.device)
    
    #print(f"action scale: {action_scale}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Networks
    actor = Actor(obs_dim, act_dim, action_scale, args.actor_hdim).to(args.device)
    qf1 = SoftQNetwork(obs_dim, act_dim, args.critic_hdim).to(args.device)
    qf2 = SoftQNetwork(obs_dim, act_dim, args.critic_hdim).to(args.device)

    base_root = f"logs/cleanrl/sac/{args.task_dir_name}"
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint 

    elif args.use_last_checkpoint:
        checkpoint_path = find_latest_checkpoint_dir(base_root)
    else:
        raise ValueError("Either --use_last_checkpoint or --checkpoint must be specified.")

    actor.load_state_dict(torch.load(os.path.join(checkpoint_path, "actor.pt")))
    qf1.load_state_dict(torch.load(os.path.join(checkpoint_path, "qf1.pt")))
    qf2.load_state_dict(torch.load(os.path.join(checkpoint_path, "qf2.pt")))

    actor.eval()
    qf1.eval()
    qf2.eval()
    
    # Logging
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    run_name_dir = f"eval_logs/cleanrl/sac/{args.task_dir_name}/{time_str}"
    summary_dir = f"{run_name_dir}/summaries"

    os.makedirs(summary_dir, exist_ok=True)

    writer = SummaryWriter(summary_dir)
    
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # reset environment
    env.unwrapped.is_Training = False
    if args.episode_length is not None:
        env.unwrapped.cfg.episode_length_s = args.episode_length
    
    wind_direc = (args.wind_direction*torch.pi/180)
    wind_speed = 5
    wind_modulo = (wind_direc + torch.pi)%(2*torch.pi) - torch.pi
    env.unwrapped._aerodynamics.update_wind(wind_direction=wind_modulo)
    env.unwrapped._aerodynamics.update_wind(wind_speed=wind_speed)

    # Start loop
    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs = obs["policy"]
    global_step = 0

    # ---- Evaluation Loop ----
    episode_cntr = 0
    # For each environment, store list of [x, y] positions
    
    num_episodes = args.num_episodes
    max_episod_length = env.unwrapped.max_episode_length
    trajectories = torch.zeros((max_episod_length, args.num_envs, 2), device=env.unwrapped.device) #[[] for _ in range(env.num_envs)]
    lift_coeff_logs = torch.zeros((max_episod_length, args.num_envs), device=env.unwrapped.device) #[[] for _ in range(env.num_envs)]
    drag_coeff_logs = torch.zeros((max_episod_length, args.num_envs), device=env.unwrapped.device) #[[] for _ in range(env.num_envs)]
    energy_logs = torch.zeros((max_episod_length, args.num_envs), device=env.unwrapped.device)
    reward_progress_logs = torch.zeros((max_episod_length, args.num_envs), device=env.unwrapped.device)
    reward_energy_logs = torch.zeros((max_episod_length, args.num_envs), device=env.unwrapped.device)
    reward_backward_logs = torch.zeros((max_episod_length, args.num_envs), device=env.unwrapped.device)  
    total_reward_logs = torch.zeros((max_episod_length, args.num_envs), device=env.unwrapped.device) 
    bearing_logs = torch.zeros((max_episod_length, args.num_envs), device=env.unwrapped.device)
    distance_logs = torch.zeros((max_episod_length, args.num_envs), device=env.unwrapped.device)

    trajectories_list = []
    lift_coeff_list = []
    drag_coeff_list = []
    goal_pos_list = []
    episode_lengths_list = []


    energy_list = []
    reward_progress_list = []
    reward_energy_list = []
    reward_backward_list = []
    total_reward_list = []
    bearing_list = []
    distance_list = []

    print(f"\n====================== Max EPISODE: {max_episod_length} ==================================\n")
    # store metrics per finished episode
    all_metrics = []
    dones = torch.zeros(args.num_envs)
    #try:
    #while simulation_app.is_running():
    current_step = 0

    while episode_cntr<num_episodes: 
    
        with torch.inference_mode():
            current_step += 1
            goal_pos =  env.unwrapped._desired_pos_w[:, :2]
            if not any(torch.equal(goal_pos, x) for x in goal_pos_list):
                goal_pos_list.append(goal_pos.clone())

            action, _ = actor.get_action(obs)
            next_obs, rew, done, trunc, info = env.step(action)

            aero_force = env.unwrapped._aerodynamic_force_b.squeeze(0)
            thruster_force = env.unwrapped._thruster_forces.squeeze(0)
            lin_speed = env.unwrapped._robot.data.root_lin_vel_b
            aoa = (180/torch.pi)*env.unwrapped._aerodynamics.angle_of_attack
            app_angle = (180/torch.pi)*env.unwrapped._aerodynamics.apparent_wind_angle # in degree
            sail = (180/torch.pi)*env.unwrapped.sail_angle
            head_w = (180/torch.pi)*env.unwrapped._robot.data.heading_w
            head_wrt_wind = torch.abs(head_w - (180/torch.pi)*env.unwrapped._aerodynamics.Beta_w)
            lift = env.unwrapped._aerodynamics.wind_lift_b
            drag = env.unwrapped._aerodynamics.wind_drag_b
            ld_ratio = torch.norm(lift, dim=-1)/torch.norm(drag, dim=-1) #torch.abs(aero_force[:, 0]/(aero_force[:, 1]+1e-6))
            robot_pos = env.unwrapped._robot.data.root_link_pos_w[:, :2]
            
            energy = env.unwrapped.energy
            episode_energy = env.unwrapped.episode_energy
            lift_coeff = env.unwrapped._aerodynamics.lift_coeff
            drag_coeff = env.unwrapped._aerodynamics.drag_coeff
            sum_angle = sail + app_angle + aoa
            desired_pos = env.unwrapped.desired_pos_b
            rew_progress = env.unwrapped.reward_progress
            rew_bearing = env.unwrapped.reward_bearing
            rew_energy = env.unwrapped.reward_energy
            rew_backward = env.unwrapped.reward_backward
            loss = env.unwrapped.loss_discrim_energy
            
            distance  = env.unwrapped.distance
            bearing = env.unwrapped.bearing
            #print(loss, rew_backward)
            loss_disc = torch.tensor([loss.item()], device=rew_backward.device) if loss is not None else torch.zeros_like(rew_energy)
            tack_wpts = env.unwrapped.tack_waypoints
            #print(f"way_pts: {env.unwrapped.sailing_mode}, bearing: {(180/torch.pi)*env.unwrapped.bearing}")
            dones = dones.to(device=trajectories.device)
            step = env.unwrapped.episode_length_buf
            episode_length = step.max().item() + 1
            if current_step%200==0:
                print(f"\nstep: {step}, episode: {episode_cntr}")
                print(f"bearing: {bearing} next_wpt: {env.unwrapped.next_tack_wpt_idx}, goal: {goal_pos}, distance: {env.unwrapped.distance}")
                #print(f"tack_wpts: {tack_wpts}, num_wpt: {env.unwrapped.tack_length}")
                #print(f"wind_direction: {(180/torch.pi)*env.unwrapped._aerodynamics.Beta_w}, sail_mode: {env.unwrapped.sailing_mode}")
            if torch.any(dones) or torch.any(trunc):
                print(f"tack_wpts: {tack_wpts}")
                
                episode_lengths_list.append(current_step)
                episode_metrics = env.unwrapped.extras["log"]
                print(f"episode: {episode_cntr} curr: {current_step} real_step: {step} dones: {dones}")
                episode_cntr += 1
                current_step = 0
                if isinstance(episode_metrics, dict):
                    all_metrics.append(episode_metrics.copy())
                
                trajectories_list.append(trajectories[:-1].clone())
                lift_coeff_list.append(lift_coeff_logs[:-1].clone())
                drag_coeff_list.append(drag_coeff_logs[:-1].clone())

                energy_list.append(energy_logs[:-1].clone())
                reward_progress_list.append(reward_progress_logs[:-1].clone())
                reward_energy_list.append(reward_energy_logs[:-1].clone())
                reward_backward_list.append(reward_backward_logs[:-1].clone())
                total_reward_list.append(total_reward_logs[:-1].clone())
                bearing_list.append(bearing_logs[:-1].clone())
                distance_list.append(distance_logs[:-1].clone())

                # Reset buffers for next episode
                trajectories.zero_()
                lift_coeff_logs.zero_()
                drag_coeff_logs.zero_()

                obs, _ = env.reset()
                if isinstance(obs, dict):
                    obs = obs["policy"]
                wind_modulo = (wind_direc + torch.pi)%(2*torch.pi) - torch.pi
                env.unwrapped._aerodynamics.update_wind(wind_direction=wind_modulo)

            #print(f"done: {dones}, type: {type(dones)}")
            alive_envs = (~dones.bool()).nonzero(as_tuple=True)[0].to(device=trajectories.device)
            #print(f"traj: {trajectories.device} alive_envs: {alive_envs.device} step: {step.device} robot_pos: {robot_pos.device}")
            trajectories[step] = torch.where(~dones.bool().unsqueeze(0), robot_pos.clone(), trajectories[step].clone())
            #lift_coeff_logs[alive_envs, step] = lift_coeff[alive_envs].float()
            
            lift_coeff_logs[step] = lift_coeff.clone().float()
            drag_coeff_logs[step] = drag_coeff.clone().float()
            energy_logs[step] = energy.clone().float()
            reward_progress_logs[step] = rew_progress.clone().float()
            reward_energy_logs[step] = rew_energy.clone().float()
            reward_backward_logs[step] = rew_backward.clone().float()
            total_reward_logs[step] = rew.clone().float()
            bearing_logs[step] = bearing.clone().float()
            distance_logs[step] = distance.clone().float()

    return all_metrics, trajectories_list, lift_coeff_list, drag_coeff_list, goal_pos_list, env, episode_lengths_list, \
                energy_list, reward_progress_list, reward_energy_list, reward_backward_list, total_reward_list, bearing_list, distance_list


if __name__ == "__main__":
    metrics_list, trajectories_list, lift_coeff_list, drag_coeff_list, goal_pos_list, env, episode_lengths_list, energy_list, \
    reward_progress_list, reward_energy_list, reward_backward_list, total_reward_list, bearing_list, distance_list = main()

    import pandas as pd
    import os
    from datetime import datetime

    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("eval_logs/sac", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving logs to: {output_dir}")

    # --- Save Trajectories ---
    traj_records = []
    for ep_idx, traj in enumerate(trajectories_list):  # (T, N, 2)
        for t in range(traj.shape[0]):
            for env_id in range(traj.shape[1]):
                traj_records.append({
                    "episode": ep_idx,
                    "time_step": t,
                    "env_id": env_id,
                    "x": traj[t, env_id, 0].item(),
                    "y": traj[t, env_id, 1].item()
                })
    pd.DataFrame(traj_records).to_csv(os.path.join(output_dir, "trajectories.csv"), index=False)

    # --- Save Aero Coefficients ---
    aero_records = []
    for ep_idx, (lift, drag) in enumerate(zip(lift_coeff_list, drag_coeff_list)):
        for t in range(lift.shape[0]):
            for env_id in range(lift.shape[1]):
                aero_records.append({
                    "episode": ep_idx,
                    "time_step": t,
                    "env_id": env_id,
                    "lift_coeff": lift[t, env_id].item(),
                    "drag_coeff": drag[t, env_id].item()
                })
    pd.DataFrame(aero_records).to_csv(os.path.join(output_dir, "aero_coeffs.csv"), index=False)

    # --- Save episode length ---
    episode_length_records = []
    for ep_idx, length in enumerate(episode_lengths_list):

        episode_length_records.append({
            "episode":ep_idx, 
            "ep_length": length
        })
    pd.DataFrame(episode_length_records).to_csv(os.path.join(output_dir, "ep_length.csv"), index=False)

    # --- Save Goals ---
    goal_records = []
    for ep_idx, goals in enumerate(goal_pos_list):  # (N, 2)
        for env_id in range(goals.shape[0]):
            goal_records.append({
                "episode": ep_idx,
                "env_id": env_id,
                "goal_x": goals[env_id, 0].item(),
                "goal_y": goals[env_id, 1].item()
            })
    pd.DataFrame(goal_records).to_csv(os.path.join(output_dir, "goals.csv"), index=False)

    # --- Save Metrics ---
    metric_records = []
    for ep_idx, m in enumerate(metrics_list):
        metric_records.append({
            "episode": ep_idx,
            "final_distance_to_goal": m["Metrics/final_distance_to_goal"],
            "consumed_energy": m["Metrics/consumed_energy"],
            "disc_prediction_mean": m["Contexts/disc_prediction_mean"]
        })
    pd.DataFrame(metric_records).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    # --- Save other Metrics Logs ---
    other_metrics_records = []
    for ep_idx in range(len(energy_list)):
        for t in range(energy_list[ep_idx].shape[0]):
            for env_id in range(energy_list[ep_idx].shape[1]):
                other_metrics_records.append({
                    "episode": ep_idx,
                    "time_step": t,
                    "env_id": env_id,
                    "energy": energy_list[ep_idx][t, env_id].item(),
                    "reward_progress": reward_progress_list[ep_idx][t, env_id].item(),
                    "reward_energy": reward_energy_list[ep_idx][t, env_id].item(),
                    "reward_backward": reward_backward_list[ep_idx][t, env_id].item(),
                    "total_reward": total_reward_list[ep_idx][t, env_id].item(),
                    "bearing": bearing_list[ep_idx][t, env_id].item(),
                    "distance": distance_list[ep_idx][t, env_id].item()
                })
    pd.DataFrame(other_metrics_records).to_csv(os.path.join(output_dir, "other_metrics.csv"), index=False)


    

    
