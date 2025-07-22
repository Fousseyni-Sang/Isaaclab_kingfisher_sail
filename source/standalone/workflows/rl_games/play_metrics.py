# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse

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
parser.add_argument("--episode_length", type=int, default=None, help="length of the episode in second.")
parser.add_argument("--wind_direction", type=float, default=180, help="direction of the true wind in degree.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
#simulation_app = app_launcher.app

"""Rest everything follows."""

# import and enable ros2 extension
import omni
from omni.isaac.core.utils.extensions import enable_extension
import rclpy

# enable ROS2 bridge extension
enable_extension("omni.isaac.ros2_bridge")

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

import numpy as np
# Create Publisher Node
"""import rclpy
from ros2_Node import RlAgentPublisher, RewardWeightSubscriber"""
import matplotlib.pyplot as plt


def main():
    """Play with RL-Games agent."""
        
    # ---- Initialize ROS2 ----
    """rclpy.init()
    ros_node = RlAgentPublisher(args_cli.num_envs)
    slider_names = ['time', 'energy', 'goal', 'wind_direct', 'desired_speed', 'wind_speed']  # Must match the names you use in the publisher
    slider_node = RewardWeightSubscriber(slider_names)"""
    #rclpy.spin(slider_node)

    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

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
    env.unwrapped.is_Training = False
    if args_cli.episode_length is not None:
        env.unwrapped.cfg.episode_length_s = args_cli.episode_length
    
    wind_direc = (args_cli.wind_direction*torch.pi/180)
    wind_speed = 5
    wind_modulo = (wind_direc + torch.pi)%(2*torch.pi) - torch.pi
    env.unwrapped._aerodynamics.update_wind(wind_direction=wind_modulo)
    env.unwrapped._aerodynamics.update_wind(wind_speed=wind_speed)
        
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.

    # ---- Evaluation Loop ----
    episode_cntr = 0
    # For each environment, store list of [x, y] positions
    
    num_episodes = 10
    max_episod_length = env.unwrapped.max_episode_length
    trajectories = torch.zeros((max_episod_length, args_cli.num_envs, 2), device=env.unwrapped.device) #[[] for _ in range(env.num_envs)]
    lift_coeff_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device) #[[] for _ in range(env.num_envs)]
    drag_coeff_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device) #[[] for _ in range(env.num_envs)]
    energy_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    reward_progress_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    reward_energy_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    reward_backward_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)  
    total_reward_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device) 
    bearing_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    distance_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)

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
    dones = torch.zeros(args_cli.num_envs)
    #try:
    #while simulation_app.is_running():
    current_step = 0
    
    while episode_cntr<num_episodes:   
        
        # run everything in inference mode
        with torch.inference_mode():
            current_step += 1
            goal_pos =  env.unwrapped._desired_pos_w[:, :2]
            # convert obs to agent format
            #print(f"\nenergy: {env.unwrapped.energy_context} \ntime: {env.unwrapped.time_context} \nwind: {env.unwrapped._aerodynamics.Beta_w}\n")
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)

            if not any(torch.equal(goal_pos, x) for x in goal_pos_list):
                goal_pos_list.append(goal_pos.clone())

            # env stepping
            obs, rew, dones, _ = env.step(actions)
            
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
                print(f"bearing: {bearing} next_wpt: {env.unwrapped.next_tack_wpt_idx}, goal: {goal_pos}, distance: {env.unwrapped.distance}")
                #print(f"tack_wpts: {tack_wpts}, num_wpt: {env.unwrapped.tack_length}")
                #print(f"wind_direction: {(180/torch.pi)*env.unwrapped._aerodynamics.Beta_w}, sail_mode: {env.unwrapped.sailing_mode}")
            if torch.any(dones):
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

                obs = env.reset()
                if isinstance(obs, dict):
                    obs = obs["obs"]
                wind_modulo = (wind_direc + torch.pi)%(2*torch.pi) - torch.pi
                env.unwrapped._aerodynamics.update_wind(wind_direction=wind_modulo)

            alive_envs = (~dones).nonzero(as_tuple=True)[0].to(device=trajectories.device)
            #print(f"traj: {trajectories.device} alive_envs: {alive_envs.device} step: {step.device} robot_pos: {robot_pos.device}")
            trajectories[step] = torch.where(~dones.unsqueeze(0), robot_pos.clone(), trajectories[step].clone())
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



        
    """finally:

        # Cleanup
        env.close()
        del env, obs, actions, rew, dones
        gc.collect()
        simulation_app.close()"""

    return all_metrics, trajectories_list, lift_coeff_list, drag_coeff_list, goal_pos_list, env, episode_lengths_list, \
                energy_list, reward_progress_list, reward_energy_list, reward_backward_list, total_reward_list, bearing_list, distance_list
   
if __name__ == "__main__":
    metrics_list, trajectories_list, lift_coeff_list, drag_coeff_list, goal_pos_list, env, episode_lengths_list, energy_list, \
    reward_progress_list, reward_energy_list, reward_backward_list, total_reward_list, bearing_list, distance_list = main()

    import pandas as pd
    import os
    from datetime import datetime

    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("eval_logs", timestamp)
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

    """print(f"Collected metrics: {len(metrics_list)} episodes")

    # Extract metrics for plotting
    final_distances = [m["Metrics/final_distance"] for m in metrics_list]
    consumed_energy = [m["Metrics/consumed_energy"] for m in metrics_list]
    disc_pred_mean = [m["Contexts/disc_prediction_mean"] for m in metrics_list]
    #print(metrics_list)

    # Plotting
    plt.figure(figsize=(12, 4))

    plt.subplot(3, 3, 1)
    plt.plot(final_distances)
    plt.title("Final Distance to Goal")
    plt.xlabel("Episode")
    plt.ylabel("Distance")

    plt.subplot(3, 3, 2)
    plt.plot(consumed_energy)
    plt.title("Consumed Energy")
    plt.xlabel("Episode")
    plt.ylabel("Energy")

    plt.subplot(3, 3, 3)
    plt.plot(disc_pred_mean)
    plt.title("Discriminator Prediction Mean")
    plt.xlabel("Episode")
    plt.ylabel("Mean Value")
    
    for ep in range(len(trajectories_list)):
        for env_idx in range(env.unwrapped.num_envs):
            x, y = trajectories_list[ep][:, env_idx, 0], trajectories_list[ep][:, env_idx, 1]
            plt.subplot(3, 3, 4)
            plt.plot(x, y)
            plt.scatter(goal_pos_list[ep][:, 0], goal_pos_list[ep][:, 1], label="goal")
            plt.title("trajectories")
            plt.xlabel("x")
            plt.ylabel("y")
            #plt.legend()

            plt.subplot(3, 3, 5)
            plt.plot(lift_coeff_list[ep][:, env_idx])
            plt.title("lift coefficient")
            plt.xlabel("Episode")
            plt.ylabel("lift coeff Values")

            plt.subplot(3, 3, 6)
            plt.plot(drag_coeff_list[ep][:, env_idx])
            plt.title("drag coefficient value")
            plt.xlabel("Episode")
            plt.ylabel("drag coeff values")
            

            print(lift_coeff_list[ep][:50])


    plt.tight_layout()
    
    #plt.show()
    plt.savefig("metrics.png")"""
    print("========================== END ! ================================")
    #simulation_app.close()
    

    
