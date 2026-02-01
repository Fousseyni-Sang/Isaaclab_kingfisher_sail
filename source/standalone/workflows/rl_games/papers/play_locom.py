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
parser.add_argument("--ros", type=bool, default=False, help="publish ros topic if True, no pub otherwise")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--episode_length", type=int, default=None, help="length of the episode in second. " \
"If None, use the default from the task config.")
parser.add_argument("--num_episode", type=int, default=10, help="number of episodes for evaluation.")

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
import rclpy
from omni.isaac.lab_tasks.utils.ros2_Node import RlAgentPublisher, RewardWeightSubscriber
#import matplotlib.pyplot as plt


def main():
    """Play with RL-Games agent."""
        
    # ---- Initialize ROS2 ----
    rclpy.init()
    ros_node = RlAgentPublisher(args_cli.num_envs)
    slider_names = ['time', 'energy', 'goal', 'wind_direct', 'desired_speed', 'wind_speed']  # Must match the names you use in the publisher
    slider_node = RewardWeightSubscriber(slider_names)
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

    acord_dir = log_dir.split("/")
    acord_dir_name = ('/'.join(acord_dir[:-3]) + "/acord/" + '/'.join(acord_dir[-2:]))
    checkpoint_name_acord = acord_dir_name + "/nn/"

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
    env.unwrapped.discr_checkpoint = checkpoint_name_acord
    env.unwrapped.discriminator_energy.load_checkpoint(checkpoint_name_acord)
    env.unwrapped.discriminator_energy.eval()
    if args_cli.episode_length is not None:
        env.unwrapped.cfg.episode_length_s = args_cli.episode_length
    
    wind_direc = (args_cli.wind_direction*torch.pi/180)
    wind_speed = args_cli.wind_speed
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
    
    num_episodes = args_cli.num_episode

    max_episod_length = env.unwrapped.max_episode_length
    trajectories = torch.zeros((max_episod_length, args_cli.num_envs, 2), device=env.unwrapped.device) #[[] for _ in range(env.num_envs)]
    lift_coeff_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device) #[[] for _ in range(env.num_envs)]
    drag_coeff_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device) #[[] for _ in range(env.num_envs)]
    energy_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    reward_progress_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    reward_energy_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    reward_backward_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    reward_aero_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    reward_acord_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)  
    total_reward_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device) 
    bearing_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    distance_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    reward_aero_force_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    aoa_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    wind_direc_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    actions_logs = torch.zeros((max_episod_length, args_cli.num_envs, 3), device=env.unwrapped.device)
    sail_angle_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    aero_force_logs = torch.zeros((max_episod_length, args_cli.num_envs, 3), device=env.unwrapped.device)
    max_aero_force_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    thruster_force_logs = torch.zeros((max_episod_length, args_cli.num_envs, 6), device=env.unwrapped.device)
    episode_energy_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    max_available_energy_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    ratio_energy_usage_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.unwrapped.device)
    acord_prediction_logs_logs = torch.zeros((max_episod_length, args_cli.num_envs, 3), device=env.unwrapped.device)
    energy_context_logs = torch.zeros((max_episod_length, args_cli.num_envs, 3), device=env.unwrapped.device)

    trajectories_list = []
    lift_coeff_list = []
    drag_coeff_list = []
    goal_pos_list = []
    energy_context_list = []
    episode_lengths_list = []

    energy_list = []
    reward_progress_list = []
    reward_energy_list = []
    reward_backward_list = []
    reward_aero_list = []
    reward_acord_list = []
    total_reward_list = []
    bearing_list = []
    distance_list = []
    reward_aero_force_list = []
    aoa_list = []
    wind_direc_list = []
    actions_list = []
    sail_angle_list = []
    aero_force_list = []
    max_aero_force_list = []
    thruster_force_list = []
    episode_energy_list = []
    max_available_energy_list = []
    ratio_energy_usage_list = []
    acord_prediction_logs_list = []

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
            energy_context = env.unwrapped.energy_context

            # convert obs to agent format
            #print(f"\nenergy: {env.unwrapped.energy_context} \ntime: {env.unwrapped.time_context} \nwind: {env.unwrapped._aerodynamics.Beta_w}\n")
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)

            if not any(torch.equal(goal_pos, x) for x in goal_pos_list):
                goal_pos_list.append(goal_pos.clone())

            """if not any(torch.equal(energy_context, x) for x in energy_context_list):
                energy_context_list.append(energy_context.clone())"""

            # env stepping
            obs, rew, dones, _ = env.step(actions)
            if actions.shape[1]==2:
                actions = torch.cat((actions, torch.zeros((1, 1), device=actions.device)), dim=-1)
            aero_force = env.unwrapped._aerodynamic_force_b.squeeze(0)
            thruster_force = env.unwrapped._thruster_forces.squeeze(0)
            lin_speed = env.unwrapped._robot.data.root_lin_vel_b
            aoa = (180/torch.pi)*env.unwrapped._aerodynamics.angle_of_attack
            app_angle = (180/torch.pi)*env.unwrapped._aerodynamics.apparent_wind_angle # in degree
            sail = (180/torch.pi)*env.unwrapped._aerodynamics.sail_angle
            head_w = (180/torch.pi)*env.unwrapped._robot.data.heading_w
            head_wrt_wind = torch.abs(head_w - (180/torch.pi)*env.unwrapped._aerodynamics.Beta_w)
            lift = env.unwrapped._aerodynamics.wind_lift_b
            drag = env.unwrapped._aerodynamics.wind_drag_b
            ld_ratio = torch.norm(lift, dim=-1)/torch.norm(drag, dim=-1) #torch.abs(aero_force[:, 0]/(aero_force[:, 1]+1e-6))
            robot_pos = env.unwrapped._robot.data.root_link_pos_w[:, :2]
            
            energy = env.unwrapped.energy
            episode_energy = env.unwrapped.episode_energy
            max_available_energy = env.unwrapped.max_available_episode_energy
            ratio_energy_usage = env.unwrapped.ratio_energy_usage
            lift_coeff = env.unwrapped._aerodynamics.lift_coeff
            drag_coeff = env.unwrapped._aerodynamics.drag_coeff
            sum_angle = sail + app_angle + aoa
            desired_pos = env.unwrapped.desired_pos_b
            rew_progress = env.unwrapped.reward_progress
            rew_bearing = env.unwrapped.reward_bearing
            rew_energy = env.unwrapped.reward_energy
            rew_backward = env.unwrapped.reward_backward
            reward_aero = env.unwrapped.reward_aero
            reward_acord = env.unwrapped.reward_acord
            loss = env.unwrapped.loss_discrim_energy
            max_aero_force = env.unwrapped.max_aero_force.squeeze(0)
            acord_prediction_logs = env.unwrapped.acord_prediction_logs
            predicted_context = acord_prediction_logs[:, 0]

            distance  = env.unwrapped.distance
            bearing = env.unwrapped.bearing
            #print(loss, rew_backward)
            loss_disc = torch.tensor([loss], device=rew_backward.device) if loss is not None else torch.zeros_like(rew_energy)
            tack_wpts = env.unwrapped.tack_waypoints
            #print(f"way_pts: {env.unwrapped.sailing_mode}, bearing: {(180/torch.pi)*env.unwrapped.bearing}")
            dones = dones.to(device=trajectories.device)
            step = env.unwrapped.episode_length_buf
            episode_length = step.max().item() + 1
            if current_step%20==0:
                #print(f"bearing: {bearing} next_wpt: {env.unwrapped.next_tack_wpt_idx}, goal: {goal_pos}, distance: {env.unwrapped.distance}")
                print(f"\nforce_aero: {aero_force}\nlift: {lift} lift_coeff: {lift_coeff} \ndrag: {drag} drag_coeff: {drag_coeff}") 
                print(f"rew_aero: {reward_aero} \nrew_prog: {rew_progress} \ncontext: {energy_context} \npredicted_context: {predicted_context}")
            if torch.any(dones) or current_step >= max_episod_length:
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
                aoa_list.append(aoa_logs[:-1].clone())
                wind_direc_list.append(wind_direc_logs[:-1].clone())
                actions_list.append(actions_logs[:-1].clone())
                sail_angle_list.append(sail_angle_logs[:-1].clone())
                aero_force_list.append(aero_force_logs[:-1].clone())
                max_aero_force_list.append(max_aero_force_logs[:-1].clone())
                thruster_force_list.append(thruster_force_logs[:-1].clone())
                episode_energy_list.append(episode_energy_logs[:-1].clone())
                max_available_energy_list.append(max_available_energy_logs[:-1].clone())
                ratio_energy_usage_list.append(ratio_energy_usage_logs[:-1].clone())
                reward_acord_list.append(reward_acord_logs[:-1].clone())
                reward_aero_list.append(reward_aero_logs[:-1].clone())
                acord_prediction_logs_list.append(acord_prediction_logs_logs[:-1].clone())
                energy_context_list.append(energy_context_logs[:-1].clone())
                



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
            
            ros_node.publish(obs, actions, rew, aero_force, thruster_force, lin_speed, aoa, app_angle, sail, 
                            head_w, head_wrt_wind, ld_ratio, robot_pos, goal_pos, energy, episode_energy, lift, drag, 
                            lift_coeff, drag_coeff, sum_angle, desired_pos, rew_progress, rew_bearing, rew_energy, 
                            rew_backward, loss_disc, tack_wpts, energy_context, predicted_context)


            lift_coeff_logs[step] = lift_coeff.clone().float()
            drag_coeff_logs[step] = drag_coeff.clone().float()
            energy_logs[step] = energy.clone().float()
            reward_progress_logs[step] = rew_progress.clone().float()
            reward_energy_logs[step] = rew_energy.clone().float()
            reward_backward_logs[step] = rew_backward.clone().float()
            total_reward_logs[step] = rew.clone().float()
            bearing_logs[step] = bearing.clone().float()
            distance_logs[step] = distance.clone().float()
            aoa_logs[step] = aoa.clone().float()
            wind_direc_logs[step] = app_angle.clone().float()
            #print(f"actions: {actions.shape} actions_logs: {actions_logs[step, 0, :].shape}")
            actions_logs[step, 0, :] = torch.tensor([actions[0, 0].clone().float(), actions[0, 1].clone().float(), 
                                            actions[0, 2].clone().float() ], device=actions_logs.device)
            sail_angle_logs[step] = sail.clone().float()
            aero_force_logs[step] = aero_force.clone().float()
            max_aero_force_logs[step] = max_aero_force.clone().float()
            thruster_force_logs[step] = thruster_force.clone().float()
            episode_energy_logs[step] = episode_energy.clone().float()
            max_available_energy_logs[step] = max_available_energy.clone().float()
            ratio_energy_usage_logs[step] = ratio_energy_usage.clone().float()
            reward_aero_logs[step] = reward_aero.clone().float()
            reward_acord_logs[step] = reward_acord.clone().float()
            acord_prediction_logs_logs[step] = acord_prediction_logs.clone().float()
            energy_context_logs[step] = energy_context.clone().float()
        
    # Cleanup
    ros_node.destroy_node()
    slider_node.destroy_node()

    rclpy.shutdown()

    return all_metrics, trajectories_list, lift_coeff_list, drag_coeff_list, goal_pos_list, env, episode_lengths_list, \
                energy_list, reward_progress_list, reward_energy_list, reward_backward_list, total_reward_list, bearing_list, \
                distance_list, aoa_list, wind_direc_list, actions_list, sail_angle_list, aero_force_list, max_aero_force_list, \
                thruster_force_list, episode_energy_list, max_available_energy_list, ratio_energy_usage_list, energy_context_list, \
                reward_acord_list, reward_aero_list, acord_prediction_logs_list
   
if __name__ == "__main__":
    metrics_list, trajectories_list, lift_coeff_list, drag_coeff_list, goal_pos_list, env, episode_lengths_list, energy_list, \
    reward_progress_list, reward_energy_list, reward_backward_list, total_reward_list, bearing_list, distance_list, aoa_list, \
    wind_direc_list, actions_list, sail_angle_list, aero_force_list, max_aero_force_list, thruster_force_list, \
         episode_energy_list, max_available_energy_list, ratio_energy_usage_list, energy_context_list, reward_acord_list, \
            reward_aero_list, acord_prediction_logs_list = main()

    import pandas as pd
    import os
    from datetime import datetime

    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = os.path.join("eval_logs/rl_games", timestamp)
    output_dir = output_dir + f"_{args_cli.wind_direction}_{args_cli.wind_speed}"
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

    # --- Save actions logs ---
    actions_records = []
    for ep_idx, actions in enumerate(actions_list):  # (T, N, D)
        for t in range(actions.shape[0]):
            for env_id in range(actions.shape[1]):
                actions_records.append({
                    "episode": ep_idx,
                    "time_step": t,
                    "env_id": env_id,
                    "action_0": actions[t, env_id, 0].item(),
                    "action_1": actions[t, env_id, 1].item(),
                    "action_2": actions[t, env_id, 2].item()
                })
    pd.DataFrame(actions_records).to_csv(os.path.join(output_dir, "actions.csv"), index=False)

    # --- Save Aero data ---
    aero_records = []
    for ep_idx, (lift, drag, aoa, app_wind_angle) in enumerate(zip(lift_coeff_list, drag_coeff_list, aoa_list, wind_direc_list)):
        for t in range(lift.shape[0]):
            for env_id in range(lift.shape[1]):
                aero_records.append({
                    "episode": ep_idx,
                    "time_step": t,
                    "env_id": env_id,
                    "lift_coeff": lift[t, env_id].item(),
                    "drag_coeff": drag[t, env_id].item(),
                    'aoa': aoa[t, env_id].item(),
                    'app_wind_angle': app_wind_angle[t, env_id].item(),
                    "sail_angle": sail_angle_list[ep_idx][t, env_id].item()
                })
    pd.DataFrame(aero_records).to_csv(os.path.join(output_dir, "aero_data.csv"), index=False)

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
                "goal_y": goals[env_id, 1].item(),
            })
    pd.DataFrame(goal_records).to_csv(os.path.join(output_dir, "goals.csv"), index=False)

    # --- Save Acord data ---
    acord_records = []
    for ep_idx, (acord_logs, context_logs) in enumerate(zip(acord_prediction_logs_list, energy_context_list)): 
        for t in range(lift.shape[0]):
            for env_id in range(acord_logs.shape[1]):
                acord_records.append({
                    "episode": ep_idx,
                    "time_step": t,
                    "env_id": env_id,
                    "predicted_context": acord_logs[t, env_id, 0].item(),
                    "actual_context": context_logs[t, env_id, 0].item(),
                    "mean": acord_logs[t, env_id, 1].item(),
                    'std': acord_logs[t, env_id, 2].item(),
                })
    pd.DataFrame(acord_records).to_csv(os.path.join(output_dir, "acord_data.csv"), index=False)


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
                    "reward_aero": reward_aero_list[ep_idx][t, env_id].item(),
                    "reward_acord": reward_acord_list[ep_idx][t, env_id].item(),
                    "total_reward": total_reward_list[ep_idx][t, env_id].item(),
                    "bearing": bearing_list[ep_idx][t, env_id].item(),
                    "distance": distance_list[ep_idx][t, env_id].item(),
                    "max_aero_force": max_aero_force_list[ep_idx][t, env_id].item(),
                    "aero_force_x": aero_force_list[ep_idx][t, env_id, 0].item(),
                    "aero_force_y": aero_force_list[ep_idx][t, env_id, 1].item(),
                    "thruster_force_1_x": thruster_force_list[ep_idx][t, env_id, 0].item(),
                    "thruster_force_2_x": thruster_force_list[ep_idx][t, env_id, 3].item(),
                    "episode_energy": episode_energy_list[ep_idx][t, env_id].item(),
                    "max_available_energy": max_available_energy_list[ep_idx][t, env_id].item(),
                    "ratio_energy_usage": ratio_energy_usage_list[ep_idx][t, env_id].item(),
                    
                
                })
    pd.DataFrame(other_metrics_records).to_csv(os.path.join(output_dir, "other_metrics.csv"), index=False)


    

    
