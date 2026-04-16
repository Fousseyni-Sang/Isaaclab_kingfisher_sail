# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext



##
# Pre-defined configs
##
import sys
sys.path.append('/home/fousseyni/asv-sawasp-fousseyni/IsaacLab/source')

import math
from my_standalone.test_rl.sailboat import SAILBOAT_ART_CFG
from omni.isaac.lab.assets import RigidObjectCollection
from omni.isaac.core.prims import RigidPrim
import my_standalone.test_rl.utils as utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
import numpy as np

def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""

    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg(physics_material = sim_utils.materials.RigidBodyMaterialCfg(static_friction = 0.0,
    dynamic_friction = 0.0))
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    
    # Create separate groups called "Origin1", "Origin2"
    # Each group will have a robot in it
    origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # Origin 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
    
    # sailboat collection of all parts 
    sailboat_cfg = SAILBOAT_ART_CFG.copy()
    sailboat_cfg.prim_path = "/World/sailboat"
    sailboat = Articulation(cfg=sailboat_cfg)


    # return the scene information
    scene_entities = {"sailboat": sailboat}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor, dc):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["sailboat"]
    physx = robot.root_physx_view
    #marker_cfg.markers["arrow"].size = (0.1, 0.1, 0.1)
    # -- goal pose
    
    
    #physx.set_masses(torch.tensor([10, 3, 3, 1]), torch.tensor([0, 1, 2, 3]))
    #physx.set_masses(data=torch.tensor([[1, 6, 6, 0.1]]), indices=torch.tensor([[0, 1, 2, 3]]))
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0   
    #physx.set_disable_gravities(torch.tensor)
    wind_direction = 90.*(math.pi/180)
    wind_speed = 3.
    epsilon = 1e-10
    sail_wing_perimeter = 2*math.pi*math.sqrt((0.15**2 + 0.03**2)/2)
    sail_wing_aire = 0.88*sail_wing_perimeter
    angle_of_attack_alpha = torch.tensor([20.*(math.pi/180)], device=sim.device) 
    asp_ratio = (0.88**2)/(0.88*0.15)
    Uw = wind_speed*torch.ones(1, device=sim.device) # wind speed of 10 meters per second
    Beta_w = wind_direction*torch.ones(1, device=sim.device) # wind angle of 80 degrees = 80*pi/180 radian
    angle = 0
    change = False
    incremente = False

    marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
    print(marker_cfg.markers)
    marker_cfg.prim_path = "/World/Command/goal_position"
    marker = VisualizationMarkers(marker_cfg)
    n = 100
    marker_translations = np.random.uniform(1, 1.5, (n, 3))

    alpha = (180*np.pi)/180
   

    # Define the number of arrows
    n = 1000

    # Generate random translations for markers within a specified range in the world frame
    marker_translations = np.random.uniform([-10, -10, 0], [10, 10, 5], (n, 3))  # Adjust bounds as needed

    # Define the wind direction (in radians)
    #alpha = wind_direction  # Assuming wind_direction is given in radians

    # Compute quaternion for rotation along the wind direction (in the world frame)
    rotation = np.array([np.cos(alpha / 2), 0, 0, np.sin(alpha / 2)])  # Rotate around Z-axis (yaw)

    # Apply the same rotation to all markers
    marker_orientations = np.tile(rotation, (n, 1))

    # Visualize markers in the world frame
    marker.visualize(translations=marker_translations, orientations=marker_orientations)


    
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        
        
        #print("Sailboat joint names:", robot.body_names)
        if count > 10:
            #print(f"[INFO]: Simulation Step count: {count}")
            
            if count %50 ==0:
                Beta_w = torch.normal(Beta_w, torch.tensor([20*torch.pi/180], device=sim.device))
            J2D =  utils.J2D(robot.data.heading_w)
            
            True_wind2D = utils.generate_true_wind_components_bis(Uw, Beta_w, robot.data.heading_w)
            ship_speed2D = robot.data.root_lin_vel_b[:, :2]

            #print(True_wind2D.shape)
            apparent_wind2D = utils.generate_apparent_wind_components_bis(True_wind2D, ship_speed2D)
            apparent_wind_angle = utils.get_apparent_wind_angle(-apparent_wind2D)
            joint_pos = robot.data.joint_pos
            
            sail_angle_target = utils.get_sail_angle(apparent_wind_angle, angle_of_attack_alpha) #+ torch.pi
            #angle_of_attack_alpha = utils.get_angle_of_attack(apparent_wind_angle, sail_angle)
            coeff_L, coeff_D = utils.generate_coeffs(angle_of_attack_alpha, asp_ratio)
           
            # Update marker positions based on wind velocity
            marker_translations += wind_speed * 0.0002

            # Handle wrapping (e.g., loop markers back into the visible region when they move too far)
            marker_translations[:, 0] = np.where(marker_translations[:, 0] > 10, -10, marker_translations[:, 0])
            marker_translations[:, 1] = np.where(marker_translations[:, 1] > 10, -10, marker_translations[:, 1])
            marker_translations[:, 2] = np.clip(marker_translations[:, 2], 0, 5)  # Keep within z bounds (0 to 5)

            # Visualize updated positions and orientations
            marker.visualize(translations=marker_translations, orientations=marker_orientations)

            lift_L, drag_D = utils.generate_force(apparent_wind2D, sail_wing_aire, coeff_L, coeff_D, density_p=1.225)
            #print(f"lift: {lift_L}, drag: {drag_D} true_wind: {True_wind2D} app_wind: {apparent_wind2D}")
            force = torch.zeros(1, 3, device=sim.device)
            #print(f"lift: {lift_L.shape} drag: {drag_D.shape}")
            force[:, 0] = lift_L*torch.cos(apparent_wind_angle) - drag_D*torch.sin(apparent_wind_angle)
            force[:, 1] = lift_L*torch.sin(apparent_wind_angle) + drag_D*torch.cos(apparent_wind_angle)
            # apply force in the y direction on the center of mass of the sail wing
            robot.set_external_force_and_torque(
                                    forces=force, 
                                    torques=epsilon*torch.ones(( 1, 3), device=sim.device), 
                                    body_ids=[3]
                                )
            # Compute control command
            error = (sail_angle_target - joint_pos + torch.pi)%(2*torch.pi) - torch.pi # Compute the error
            control_command = robot.data.joint_pos + error  # Apply proportional control

            print(robot.data.body_pos_w[:, [1, 3], :2])

            # Proportional control gain
            k_p = 0.1  # Adjust this value based on the simulation's requirement

            robot.set_joint_position_target(target=control_command, joint_ids=[0])


            # -- write data to sim
            robot.write_data_to_sim()

            # Update buffers
            robot.update(sim_dt)
            
        # Perform step

        sim.step()
        # Increment counter
        
        """if count == 602:
            break
            sys.exit(1)"""
        count += 1

def main():
    """Main function."""

    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    dc=_dynamic_control.acquire_dynamic_control_interface()

    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins, dc)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()