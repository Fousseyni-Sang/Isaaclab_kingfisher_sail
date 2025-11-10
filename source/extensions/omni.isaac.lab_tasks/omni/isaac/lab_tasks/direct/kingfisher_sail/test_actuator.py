# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.physics.foil_dynamics import FoilDynamicsCfg, FoilDynamics
from omni.isaac.lab.actuator_force.foil_actuator_force import FoilActuator, FoilActuatorCfg
import torch
import matplotlib.pyplot as plt
import numpy as np

dir = "/mnt/gpu_storage/zrr/fsangare/asv-sawasp-fousseyni/Isaaclab_kingfisher_sail/Test_plot/rudder_actuator/"
prefix = "rudder_"
import os
os.makedirs(dir, exist_ok=True)
#=====================================================================================================================#
# Rudder Hydrodynamics: flow is the water
rudder_hydrodyn_cfg: FoilDynamicsCfg = FoilDynamicsCfg()
rudder_hydrodyn_cfg.flow_density = 997.0 # water density kg/m^3
rudder_hydrodyn_cfg.foil_span = 1
rudder_hydrodyn_cfg.foil_chord = 0.2
rudder_hydrodyn_cfg.flow_direction = -179*torch.pi/180
rudder_hydrodyn_cfg.flow_speed = 0.5
rudder_hydrodyn_cfg.angle_of_attack = 20*torch.pi/180
rudder_hydrodyn_cfg.min_upflow_angle = 45*torch.pi/180
rudder_hydrodyn_cfg.max_downflow_angle = 140*torch.pi/180
rudder_hydrodyn_cfg.Reynold = 1000000  # based on flow_speed, chord, density and viscosity
device = 'cpu'

actuator_cfg: FoilActuatorCfg = FoilActuatorCfg()
actuator_cfg.cmd_lower_range = -1.0
actuator_cfg.cmd_upper_range = 1.0
actuator_cfg.command_rate = (actuator_cfg.cmd_upper_range - actuator_cfg.cmd_lower_range) / 2.0
actuator_cfg.resolution = 1.8  # degrees
actuator_cfg.precision = 0.05  # radians
actuator_cfg.scale_joint_pos = torch.pi  # radians per command unit
actuator_cfg.pos_from_com = (0.0, 0.0, 0.0)  # meters

num_envs = 1
#=====================================================================================================================#
rudder_hydrodyn = FoilDynamics(num_envs=num_envs, device=device, cfg=rudder_hydrodyn_cfg, naca_profile="0012")
rudder_actuator = FoilActuator(num_envs=num_envs, dynamics=rudder_hydrodyn, dt=0.05, cfg=actuator_cfg)

target_cmds = torch.tensor([[1.0]], device=device)  
robot_heading = torch.tensor([[0.0]], device=device)
robot_vel_b = torch.tensor([[0.0, 0.0, 0.0]], device=device)  # moving forward at 0.5 m/s
rudder_hydrodyn.init_flow_vector(
    robot_vel_b[:, 0:2], robot_heading
)
forces_list = []
torque_list = []
angles_list = []
cmd_list = []
aoa_list = []
current_joint_pos = torch.tensor([[0.0]], device=device)
iteration = 0
while rudder_actuator.current_cmd.abs() < torch.pi*target_cmds.abs():
    rudder_actuator.update_joint_cmd(current_joint_pos, target_cmds)
    rudder_actuator.update_forces(robot_heading, robot_vel_b)
    rudder_angle = rudder_actuator.get_joint_positions()
    forces_torques = rudder_actuator.get_forces()

    cmd_list.append(rudder_actuator.current_cmd.item())
    forces_list.append((forces_torques[0,0].item(), forces_torques[0,1].item(), forces_torques[0,2].item()))
    torque_list.append((forces_torques[0,3].item(), forces_torques[0,4].item(), forces_torques[0,5].item()))
    angles_list.append((180/np.pi)*rudder_angle.item())
    aoa_list.append((180/np.pi)*rudder_hydrodyn.angle_of_attack.item())

    current_joint_pos = rudder_angle.clone()

    #print(f"\nIteration {iteration}:")
    #print(f"Current cmd: {rudder_actuator.get_current_cmd.item():.4f}, Rudder Angle (deg): {rudder_angle.item()*180/np.pi:.2f}")

    iteration += 1


print("Rudder Actuator Test Completed.")
#print(cmd_list)
# Plot Rudder Actuator Response
forces_x, forces_y, forces_z = zip(*forces_list)
plt.figure()
plt.scatter(aoa_list, forces_x, label='Force X')
plt.scatter(aoa_list, forces_y, label='Force Y')
plt.plot(aoa_list, forces_z, label='Force Z')
plt.xlabel('angle of attack (radian)')
plt.ylabel('Forces (N)')
plt.title('Rudder Actuator Forces vs aoa')
plt.legend()
plt.savefig(dir+prefix+"forces_vs_aoa.png")

plt.figure()
plt.plot(cmd_list, label='Rudder cmd')
plt.hlines(y=torch.pi*target_cmds.item(), xmin=0*target_cmds, xmax=len(cmd_list), color='r', linestyle='--', label='Target cmd' )
plt.xlabel('time step')
plt.ylabel('cmd value')
plt.title('Command Response Over Time')
plt.legend()
plt.savefig(dir+prefix+"command_response.png")  

plt.figure()
plt.plot(angles_list, label='Rudder angle (rad)')
plt.xlabel('time step')
plt.ylabel('angle value')
plt.title('angle Over Time')
plt.legend()
plt.savefig(dir+prefix+"rudder_angle.png")  

plt.figure()
plt.plot(aoa_list, label='angle of attack(rad)')
plt.xlabel('time step')
plt.ylabel('angle of attack value')
plt.title('angle of attack Over Time')
plt.legend()
plt.savefig(dir+prefix+"angle_of_attack.png")  

    
