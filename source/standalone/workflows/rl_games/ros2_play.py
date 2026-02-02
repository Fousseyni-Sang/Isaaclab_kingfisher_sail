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


# Create Publisher Node
import rclpy
from omni.isaac.lab_tasks.utils.my_config.ros2_Node import RlAgentPublisher, RewardWeightSubscriber


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


    while simulation_app.is_running():
        rclpy.spin_once(slider_node, timeout_sec=0.0)
        
        time_context = slider_node.reward_weights["time"]
        energy_context = slider_node.reward_weights["energy"]
        reset_env = energy_context > 1.5 or time_context > 1.5
        desired_speed = slider_node.reward_weights["desired_speed"]
        wind_direc = slider_node.reward_weights["wind_direct"]*(torch.pi/180)
        wind_speed = slider_node.reward_weights["wind_speed"] if slider_node.reward_weights["wind_speed"] else 1e-6
        goal_pos = slider_node.reward_weights["goal"]
        env.unwrapped.cfg.min_target_distance *= goal_pos
        wind_modulo = (wind_direc + torch.pi)%(2*torch.pi) - torch.pi
        env.unwrapped._sail_aerodynamics.update_wind(wind_direction=wind_modulo)
        env.unwrapped.energy_context[:] = energy_context
        env.unwrapped.time_context[:] = time_context
        #env.unwrapped.desired_speed_b[:] = desired_speed
        env.unwrapped.corridor_width[:] = desired_speed
        env.unwrapped._sail_aerodynamics.update_wind(wind_speed=wind_speed)
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            #print(f"\nenergy: {env.unwrapped.energy_context} \ntime: {env.unwrapped.time_context} \nwind: {env.unwrapped._sail_aerodynamics.Beta_w}\n")
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            
            # env stepping
            obs, rew, dones, _ = env.step(actions)
            aero_force = env.unwrapped._aerodynamic_force_b.squeeze(0)
            thruster_force = env.unwrapped._thruster_forces.squeeze(0)
            lin_speed = env.unwrapped._robot.data.root_lin_vel_b
            aoa = (180/torch.pi)*env.unwrapped._sail_aerodynamics.angle_of_attack
            app_angle = (180/torch.pi)*env.unwrapped._sail_aerodynamics.apparent_wind_angle # in degree
            sail = (180/torch.pi)*env.unwrapped.sail_angle
            head_w = (180/torch.pi)*env.unwrapped._robot.data.heading_w
            head_wrt_wind = torch.abs(head_w - (180/torch.pi)*env.unwrapped._sail_aerodynamics.Beta_w)
            lift = env.unwrapped._sail_aerodynamics.wind_lift_b
            drag = env.unwrapped._sail_aerodynamics.wind_drag_b
            ld_ratio = torch.norm(lift, dim=-1)/torch.norm(drag, dim=-1) #torch.abs(aero_force[:, 0]/(aero_force[:, 1]+1e-6))
            robot_pos = env.unwrapped._robot.data.root_link_pos_w[:, :2]
            goal_pos =  env.unwrapped._desired_pos_w[:, :2]
            energy = env.unwrapped.energy
            episode_energy = env.unwrapped.episode_energy
            lift_coeff = env.unwrapped._sail_aerodynamics.lift_coeff
            drag_coeff = env.unwrapped._sail_aerodynamics.drag_coeff
            sum_angle = sail + app_angle + aoa
            desired_pos = env.unwrapped.desired_pos_b
            rew_progress = env.unwrapped.reward_progress
            rew_bearing = env.unwrapped.reward_bearing
            rew_energy = env.unwrapped.reward_energy
            rew_backward = env.unwrapped.reward_backward
            loss = env.unwrapped.loss_discrim_energy
            #print(loss, rew_backward)
            loss_disc = torch.tensor([loss.item()], device=rew_backward.device) if loss is not None else torch.zeros_like(rew_energy)
            tack_wpts = env.unwrapped.tack_waypoints

            if reset_env:
                env.reset()
                

            ros_node.publish(obs, actions, rew, aero_force, thruster_force, lin_speed, aoa, app_angle, sail, 
                            head_w, head_wrt_wind, ld_ratio, robot_pos, goal_pos, energy, episode_energy, lift, drag, 
                            lift_coeff, drag_coeff, sum_angle, desired_pos, rew_progress, rew_bearing, rew_energy, 
                            rew_backward, loss_disc, tack_wpts)


            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    
    # Cleanup
    ros_node.destroy_node()
    slider_node.destroy_node()

    rclpy.shutdown()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    
    # close sim app
    simulation_app.close()
