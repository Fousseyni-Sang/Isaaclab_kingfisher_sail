# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-KingfisherSail-Direct-High-SP-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--algo", type=str, default="ppo", help="name of the algorith to use in SB3: (ppo, ddpg, sac)")
parser.add_argument("--ros", action="store_true", default=False, help="Enable ROS2 publishing.")
parser.add_argument("--ros_publish_interval", type=int, default=10, help="ROS publish interval in steps.") 
parser.add_argument("--num_episode", type=int, default=10, help="Number of episodes for evaluation.") 

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
import numpy as np
import os
import torch

from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.vec_env import VecNormalize

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import rclpy 
from omni.isaac.lab_tasks.utils.my_utils.ros2_Node import RewardWeightSubscriber, DynamicsRlAgentPublisher

def main():
    """Play with stable-baselines agent."""

    # ROS init (optional) 

    if args_cli.ros: 
        rclpy.init() 
        dyn_ros_node = DynamicsRlAgentPublisher(args_cli.num_envs) 
    else: 
        dyn_ros_node = None 

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    if args_cli.algo=="sac":
        algo = SAC
        agent_cfg = agent_cfg["sac"]
    elif args_cli.algo=="ddpg":
        algo = DDPG
        agent_cfg = agent_cfg["ddpg"]
    else:
        algo = PPO
        agent_cfg = agent_cfg["ppo"]

    

    # directory for logging into
    log_root_path = os.path.join("logs", "sb3", args_cli.task, args_cli.algo)
    log_root_path = os.path.abspath(log_root_path)
    # check checkpoint is valid
    if args_cli.checkpoint is None:
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint)
    else:
        checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(checkpoint_path)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)

    # normalize environment (if needed)
    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent = algo.load(checkpoint_path, env, print_system_info=True)

    # reset environment
    obs = env.reset()
    timestep = 0
    per_env_goal = [None for _ in range(args_cli.num_envs)] 
    num_episodes_target = args_cli.num_episode 
    episodes_finished = 0 

    all_episode_logs = []   # list of dicts: { "episode": i, "env_id": j, "steps": tensor }

    per_env_episode_lengths = [] 
    all_metrics = [] 
    env.unwrapped.is_Training = False # to control whether env.extras["info"] is updated at each step
    # simulate environment
    while simulation_app.is_running() and episodes_finished < num_episodes_target:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions, _ = agent.predict(obs, deterministic=True)
            # env stepping
            obs, rew, dones, extras = env.step(actions)

            #print(obs_pub)
            obs_pub = []
            robot_pos = [] #extras["info"]["robot_pos_w"][..., :2] # (N, 2) 
            lift_coeff = [] #extras["info"]["lift_coeff"] # (N,) 
            drag_coeff = [] #extras["info"]["drag_coeff"] # (N,) 
            energy = [] #extras["info"]["energy"] # (N,) 
            episode_energy = [] #extras["info"]["episode_energy"] # (N,) 
            rew_progress = [] #extras["info"]["reward_progress"] # (N,) 
            rew_energy = [] #extras["info"]["reward_energy"] # (N,) 
            rew_backward = [] #extras["info"]["reward_backward"] # (N,) 
            rew_aero = [] #extras["info"]["reward_aero"] # (N,) 
            rew_acord = [] #extras["info"]["reward_acord"] # (N,) 
            bearing = [] #extras["info"]["bearing"] # (N,) 
            distance = [] #extras["info"]["distance"] # (N,) 
            lin_vel_b = [] #extras["info"]["lin_vel_b"] # (N, 3)
            ang_vel_b = [] #extras["info"]["ang_vel_b"] # (N, 3)
            aoa = [] #extras["info"]["aoa"] # (N,) 
            app_wind_angle = [] #extras["info"]["app_wind_angle"] # (N,) 
            true_wind_angle = [] #extras["info"]["true_wind_angle"] # (N,) 
            sail_angle = [] #extras["info"]["sail_angle"] # (N,) 
            aero_force = [] #extras["info"]["aero_force"] # (N, 6) 
            max_aero_force = [] #extras["info"]["max_aero_force"] # (N,) 
            thruster_force = [] #extras["info"]["thruster_force"] # (N, 6) 
            norm_error_lin = [] #extras["info"]["norm_error_lin"] # (N,) 
            norm_error_ang = [] #extras["info"]["norm_error_ang"] # (N,) 
            goal_pos = [] #extras["info"]["goal_pos"] # (N, 2) 
            heading_w = [] #extras["info"]["heading_w"] # (N,)
            lift_drag_ratio = [] #extras["info"]["lift_drag_ratio"] # (N,)
            lift_force_b = [] #extras["info"]["lift_force_b"] # (N, 3)
            drag_force_b = [] #extras["info"]["drag_force_b"] # (N,
            norm_error_cat=[] #torch.cat( [norm_error_lin, norm_error_ang.reshape(-1, 1)], dim=-1 )
            desired_wrench_b = [] #extras["info"]["desired_wrench_b"] # (N, 6)
            # Save goal per env once 
            for env_id in range(args_cli.num_envs): 
                obs_pub.append(np.concatenate([obs[key][env_id] for key in obs])) # (N, obs_dim)
                
                robot_pos.append(extras[env_id]["info"]["robot_pos_w"][:2]) # (2) 
                lift_coeff.append(extras[env_id]["info"]["lift_coeff"]) # (N,) 
                drag_coeff.append(extras[env_id]["info"]["drag_coeff"]) # (N,) 
                energy.append(extras[env_id]["info"]["energy"]) # (N,) 
                episode_energy.append(extras[env_id]["info"]["episode_energy"]) # (N,) 
                rew_progress.append(extras[env_id]["info"]["reward_progress"]) # (N,) 
                rew_energy.append(extras[env_id]["info"]["reward_energy"]) # (N,) 
                rew_backward.append(extras[env_id]["info"]["reward_backward"]) # (N,) 
                rew_aero.append(extras[env_id]["info"]["reward_aero"]) # (N,) 
                rew_acord.append(extras[env_id]["info"]["reward_acord"]) # (N,) 
                bearing.append(extras[env_id]["info"]["bearing"]) # (N,) 
                distance.append(extras[env_id]["info"]["distance"]) # (N,) 
                aoa.append(extras[env_id]["info"]["aoa"]) # (N,) 
                app_wind_angle.append(extras[env_id]["info"]["app_wind_angle"]) # (N,) 
                true_wind_angle.append(extras[env_id]["info"]["true_wind_angle"]) # (N,) 
                sail_angle.append(extras[env_id]["info"]["sail_angle"]) # (N,) 
                aero_force.append(extras[env_id]["info"]["aero_force"]) # (N, 6) 
                max_aero_force.append(extras[env_id]["info"]["max_aero_force"]) # (N,) 
                thruster_force.append(extras[env_id]["info"]["thruster_force"]) # (N, 6) 
                norm_error_lin.append(extras[env_id]["info"]["norm_error_lin"]) # (N,) 
                norm_error_ang.append(extras[env_id]["info"]["norm_error_ang"]) # (N,) 
                lin_vel_b.append(extras[env_id]["info"]["lin_vel_b"]) # (N, 3)
                ang_vel_b.append(extras[env_id]["info"]["ang_vel_b"]) # (N, 3)
                goal_pos.append(extras[env_id]["info"]["goal_pos"]) # (N, 2) 
                heading_w.append(extras[env_id]["info"]["heading_w"]) # (N,)
                lift_force_b.append(extras[env_id]["info"]["lift_force_b"]) # (N, 3)
                drag_force_b.append(extras[env_id]["info"]["drag_force_b"]) # (N, 3) 
                desired_wrench_b.append(extras[env_id]["info"]["desired_wrench_b"]) # (N, 6)   
                norm_error_cat.append(extras[env_id]["info"]["norm_error_cat" ]) # (N, 3)
                
                if per_env_goal[env_id] is None: 
                    per_env_goal[env_id] = goal_pos[env_id].detach().cpu().clone() 
        
                if dones[env_id]: 
                    # --------------------------------------------------------- 
                    # Store the entire episode in a global list 
                    # --------------------------------------------------------- 
                    all_episode_logs.append({ 
                        "episode": episodes_finished, 
                        "env_id": env_id, 
                        }) 
                    # --------------------------------------------------------- 
                    # Metrics from env.extras["log"] 
                    # --------------------------------------------------------- 
                    if "log" in extras[env_id]: 
                        metrics = extras[env_id]["log"] 
                        if isinstance(metrics, dict): 
                            metrics_copy = metrics.copy() 
                            metrics_copy["episode"] = episodes_finished 
                            metrics_copy["env_id"] = env_id 
                            all_metrics.append(metrics_copy) 
                    print(f"[INFO] Episode {episodes_finished} finished (env {env_id}) ep_length: {env.unwrapped.episode_length_buf[env_id]}") 
                    episodes_finished += 1 

            # ROS publish (throttled) 
            
            if args_cli.ros: # and (step_idx % args_cli.ros_publish_interval == 0): 
                dyn_ros_node.publish( 
                    obs=obs_pub, 
                    actions=actions, 
                    total_rew=rew, 
                    aero_force=aero_force, 
                    thruster_force=thruster_force,
                    lin_vel_b=torch.stack(lin_vel_b), 
                    angle_of_attack=aoa, 
                    app_flow_angle=app_wind_angle, 
                    true_flow_angle=true_wind_angle, 
                    sail_angle=sail_angle, 
                    heading_w=torch.stack(heading_w), 
                    robot_pos_w=robot_pos, 
                    energy=energy, 
                    episode_energy=episode_energy, 
                    lift_force_b=torch.stack(lift_force_b), 
                    drag_force_b=torch.stack(drag_force_b), 
                    lift_coeff=lift_coeff, 
                    drag_coeff=drag_coeff, 
                    rew_progress=rew_progress, 
                    rew_energy=rew_energy, 
                    rew_backward=rew_backward, 
                    desired_wrench_b=desired_wrench_b, 
                    ang_speed_b=ang_vel_b, 
                    norm_error_cat=norm_error_cat, 
                    rew_aero=rew_aero, 
                    distance=distance, 
                    goal_pos=goal_pos,
                    )

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
