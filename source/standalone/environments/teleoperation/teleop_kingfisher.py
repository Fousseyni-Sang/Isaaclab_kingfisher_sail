# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch

import omni.log

from omni.isaac.lab.devices import Se3GamepadKingfisher, Se3Keyboard, Se3SpaceMouse
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils import parse_env_cfg


# Create Publisher Node
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class RlAgentPublisher(Node):
    def __init__(self):
        super().__init__("rl_agent_publisher")
        self.obs_publisher = self.create_publisher(Float32MultiArray, "rl_observations", 10)
        self.act_publisher = self.create_publisher(Float32MultiArray, "rl_actions", 10)
        self.rew_publisher = self.create_publisher(Float32MultiArray, "rl_rewards", 10)
        self.aero_force_publisher = self.create_publisher(Float32MultiArray, "aero_force", 10)

    def publish_obs(self, obs):
        
        msg = Float32MultiArray()
        msg.data = obs.cpu().numpy().flatten().tolist()
        self.obs_publisher.publish(msg)

    def publish_act(self, act):
        msg = Float32MultiArray()
        msg.data = act.cpu().numpy().flatten().tolist()
        self.act_publisher.publish(msg)

    def publish_rew(self, rew):
        msg = Float32MultiArray()
        msg.data = rew.cpu().numpy().flatten().tolist()
        self.rew_publisher.publish(msg)
    def publish_aero_force(self, aero_force):
        msg = Float32MultiArray()
        msg.data = aero_force.cpu().numpy().flatten().tolist()
        self.aero_force_publisher.publish(msg)

def pre_process_actions(delta_pose: torch.Tensor) -> torch.Tensor:
    """Pre-process actions for the environment."""
    thrust_right = delta_pose[:, 0:1]

    thrust_left = delta_pose[:, 2:3]

    if delta_pose[:, 5:6]>=0.01:
        sail_angle = delta_pose[:, 5:6] 
    else: 
        sail_angle = delta_pose[:, 1:2]
    actions = torch.cat((thrust_right, thrust_left, sail_angle), dim=1)
    #print(delta_pose)
    return actions


def main():

     # ---- Initialize ROS2 ----
    rclpy.init()
    ros_node = RlAgentPublisher()

    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # modify configuration
    env_cfg.episode_length_s = 100000000000
    """env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        omni.log.warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )"""
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.005 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3GamepadKingfisher(
            pos_sensitivity=args_cli.sensitivity, rot_sensitivity= args_cli.sensitivity
        )
        
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'.")
    # add teleoperation key for env reset
    #teleop_interface.add_callback("L", env.reset) # It was previously "L" instead of B
    #teleop_interface.add_callback()
    # print helper for keyboard
    #print(teleop_interface)

    # reset environment
    env.reset()
    teleop_interface.reset()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
            delta_pose = delta_pose.astype("float32")
            # convert to torch
            delta_pose = torch.tensor(delta_pose, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            # pre-process actions
            actions = pre_process_actions(delta_pose)
            #print(actions)
            # apply actions

            # env stepping
            obs, rew, dones, _, _ = env.step(actions)

            # ---- Publish observations and actions to ROS2 ----
            ros_node.publish_obs(obs["policy"])
            ros_node.publish_act(actions)
            ros_node.publish_rew(rew)
            ros_node.publish_aero_force(env.unwrapped._aerodynamic_force.squeeze(0))
            
    # Cleanup
    ros_node.destroy_node()
    rclpy.shutdown()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
