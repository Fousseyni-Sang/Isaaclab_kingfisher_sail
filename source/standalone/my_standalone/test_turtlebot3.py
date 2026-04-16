# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/02_scene/create_scene_heron.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

torch.set_printoptions(precision=2, sci_mode=False)

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_assets.turtlebot3 import TURTLEBOT3_BURGER_CFG
import numpy as np


import torch

@configclass
class HeronSceneCfg(InteractiveSceneCfg):
    """Configuration for a heron scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    turtlebot3: ArticulationCfg = TURTLEBOT3_BURGER_CFG.copy()
    turtlebot3.prim_path = "/World/Robot"

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    # Extract scene entities

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    robot = scene["turtlebot3"]
    
    
    # Get the link ids
    base_link_id, _ = robot.find_bodies("a__namespace_base_link")
    
    left_wheel_joint_id, _ = robot.find_joints("a__namespace_wheel_left_joint")
    right_wheel_joint_id, _ = robot.find_joints("a__namespace_wheel_right_joint")
    print(f"base_link_id: {base_link_id}, left_wheel_joint_id: {left_wheel_joint_id}, right_wheel_joint_id: {right_wheel_joint_id}")
    count = 0
    while simulation_app.is_running():
        # Reset
        if count == 0:
            
            # reset counter
            print("==========")
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()

            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()

            #joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")


        robot.set_joint_velocity_target(target=torch.tensor([[10.0, 10.0]], device=joint_vel.device), joint_ids=[left_wheel_joint_id[0], right_wheel_joint_id[0]])

        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([4.0, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = HeronSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
