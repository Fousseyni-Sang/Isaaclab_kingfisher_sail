# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from ..rl_games.ros2_Node import RlAgentPublisher, RewardWeightSubscriber
from omni.isaac.core.utils.extensions import enable_extension
import rclpy

# enable ROS2 bridge extension
enable_extension("omni.isaac.ros2_bridge")

def main():
    """Play with RSL-RL agent."""

    # ---- Initialize ROS2 ----
    rclpy.init()
    ros_node = RlAgentPublisher(args_cli.num_envs)
    slider_names = ['time', 'energy', 'goal', 'wind_direct', 'desired_speed', 'wind_speed']  # Must match the names you use in the publisher
    slider_node = RewardWeightSubscriber(slider_names)
    #rclpy.spin(slider_node)

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )
    # reset environment
    env.unwrapped.is_Training = False

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        rclpy.spin_once(slider_node, timeout_sec=0.0)
        
        time_context = slider_node.reward_weights["time"]
        energy_context = slider_node.reward_weights["energy"]
        reset_env = energy_context > 1.5 or time_context > 1.5
        desired_speed = slider_node.reward_weights["desired_speed"]
        wind_direc = slider_node.reward_weights["wind_direct"]*(torch.pi/180)
        wind_speed = slider_node.reward_weights["wind_speed"] if slider_node.reward_weights["wind_speed"] else 1e-6
        wind_modulo = (wind_direc + torch.pi)%(2*torch.pi) - torch.pi
        env.unwrapped._aerodynamics.update_wind(wind_direction=wind_modulo)
        env.unwrapped.energy_context[:] = energy_context
        env.unwrapped.time_context[:] = time_context
        env.unwrapped.desired_speed_b[:] = desired_speed
        env.unwrapped._aerodynamics.update_wind(wind_speed=wind_speed)

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rew, _, _ = env.step(actions)

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
            goal_pos =  env.unwrapped._desired_pos_w[:, :2]
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
            #print(loss, rew_backward)
            loss_disc = torch.tensor([loss.item()], device=rew_backward.device) if loss is not None else torch.zeros_like(rew_energy)

            if reset_env:
                env.reset()

            ros_node.publish(obs, actions, rew, aero_force, thruster_force, lin_speed, aoa, app_angle, sail, 
                             head_w, head_wrt_wind, ld_ratio, robot_pos, goal_pos, energy, episode_energy, lift, drag, 
                             lift_coeff, drag_coeff, sum_angle, desired_pos, rew_progress, rew_bearing, rew_energy, rew_backward, loss_disc)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
    
    # close the ROS2 node
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
