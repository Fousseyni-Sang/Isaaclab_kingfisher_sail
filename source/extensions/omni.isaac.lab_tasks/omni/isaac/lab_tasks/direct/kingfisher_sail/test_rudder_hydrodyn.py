# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import omni
from omni.isaac.lab.physics.foil_dynamics import FoilDynamicsCfg, FoilDynamics
import torch
import matplotlib.pyplot as plt
import numpy as np

def rudder_config():
    
    # Rudder Hydrodynamics
    cfg: FoilDynamicsCfg = FoilDynamicsCfg()
    cfg.flow_density = 997.0
    cfg.foil_span = 1
    cfg.foil_chord = 0.2
    cfg.flow_direction = 30*torch.pi/180
    cfg.flow_speed = 0.05
    cfg.angle_of_attack = 20*torch.pi/180
    cfg.min_upflow_angle = 45*torch.pi/180
    cfg.max_downflow_angle = 140*torch.pi/180
    cfg.Reynold = 500000  # based on flow_speed, chord, density and viscosity
    return cfg

# Rudder Hydrodynamics: flow is the water
rudder_hydrodyn_cfg: FoilDynamicsCfg = rudder_config()
"""rudder_hydrodyn_cfg.flow_density = 997.0 # water density kg/m^3
rudder_hydrodyn_cfg.foil_span = 1
rudder_hydrodyn_cfg.foil_chord = 0.2
rudder_hydrodyn_cfg.flow_direction = -179*torch.pi/180
rudder_hydrodyn_cfg.flow_speed = 0.5
rudder_hydrodyn_cfg.angle_of_attack = 20*torch.pi/180
rudder_hydrodyn_cfg.min_upflow_angle = 45*torch.pi/180
rudder_hydrodyn_cfg.max_downflow_angle = 140*torch.pi/180
rudder_hydrodyn_cfg.Reynold = 1000000  # based on flow_speed, chord, density and viscosity"""
device = 'cpu'


min_aoa = -180
max_aoa = 180
num_envs = int(abs(min_aoa) + abs(max_aoa))
#=====================================================================================================================#
rudder_hydrodyn = FoilDynamics(num_envs=num_envs, device=device, cfg=rudder_hydrodyn_cfg)
angle_of_attack_degree = torch.linspace(min_aoa, max_aoa, num_envs, device=device)
angle_of_attack_radian = (torch.pi/180)*angle_of_attack_degree


dir = "/mnt/gpu_storage/zrr/fsangare/asv-sawasp-fousseyni/Isaaclab_kingfisher_sail/Test_plot/rudder_hydrodynamics/"
prefix = "rudder_"
import os
os.makedirs(dir, exist_ok=True)


#=====================================================================================================================#
# Compute hydrodynamic forces 
#Uw = 0.1*torch.ones_like(angle_of_attack)  # flow speed
#Beta_w = 0.0*torch.ones_like(angle_of_attack)  # flow direction
ship_heading_w = (torch.pi/180)*0.0*torch.ones_like(angle_of_attack_radian)  # ship heading in world
ship_lin_vel2D = torch.tensor([[0.0, 0.0]], device=device).repeat(angle_of_attack_radian.shape[0], 1)  # ship linear velocity in 2D
true_flow_vel2D = rudder_hydrodyn.generate_true_flow_components(rudder_hydrodyn.Uw, rudder_hydrodyn.Beta_w, ship_heading_w)
apparent_flow_vel2D = rudder_hydrodyn.generate_apparent_flow_components(true_flow_vel2D, ship_lin_vel2D)
apparent_flow_angle = rudder_hydrodyn.get_apparent_flow_angle(apparent_flow_vel2D)
foil_angl = rudder_hydrodyn.get_foil_angle(apparent_flow_angle, angle_of_attack_radian)

hydro_forces = rudder_hydrodyn.compute_flow_effect(rudder_hydrodyn.Uw, rudder_hydrodyn.Beta_w, ship_heading_w, ship_lin_vel2D, angle_of_attack_radian, foil_angl)
norm_hydro_forces = torch.norm(hydro_forces, dim=1)
lift, drag = rudder_hydrodyn.flow_lift_b, rudder_hydrodyn.flow_drag_b

cl = rudder_hydrodyn.lift_coeff
cd = rudder_hydrodyn.drag_coeff

magn_lift, magn_drag = rudder_hydrodyn.lift_magnitude, rudder_hydrodyn.drag_magnitude

# Open Aerodynamics figure:  xdg-open /tmp/cl_cd_aoa.png
plt.figure()
plt.plot(angle_of_attack_degree.cpu(), cl.cpu(), '--', c='g', label="cl")
plt.plot(angle_of_attack_degree.cpu(), cd.cpu(), '--', c='y', label="cd")
plt.xlabel("angle of attack (degree)")
plt.ylabel("cl and cd")
plt.legend()
plt.title("Lift and Drag Coefficients vs Angle of Attack")
plt.savefig(dir+prefix+"cl_cd_aoa.png")

plt.figure()
plt.plot(angle_of_attack_degree.cpu(), (cl/cd).cpu(), c='g', label="cl/cd")
plt.xlabel("angle of attack (degree)")
plt.ylabel("cl/cd")
plt.legend()
plt.title("Lift-to-Drag Ratio vs Angle of Attack")
plt.savefig(dir+prefix+"cl_cd_ratio.png")

plt.figure()
plt.plot(cd.cpu(), cl.cpu(), c='grey', label="cl vs cd")
plt.xlabel("cd")
plt.ylabel("cl")
plt.legend()
plt.title("Lift Coefficient vs Drag Coefficient")
plt.savefig(dir+prefix+"cl_vs_cd.png")

plt.figure()
plt.plot(angle_of_attack_degree.cpu(), lift.cpu(), '--', c='g', label="Lift")
plt.plot(angle_of_attack_degree.cpu(), drag.cpu(), '--', c='y', label="Drag")
plt.xlabel("angle of attack (degree)")
plt.ylabel("forces (N)")
plt.legend()
plt.title("Vector components of lift and drag")
plt.savefig(dir+prefix+"vec_lift_drag_aoa.png")

plt.figure()
plt.plot(angle_of_attack_degree.cpu(), hydro_forces[:, 0].cpu(), c='g', label="Force X_b")
plt.plot(angle_of_attack_degree.cpu(), hydro_forces[:, 1].cpu(), c='r', label="Force Y_b")
plt.xlabel("angle of attack (degree)")
plt.ylabel("forces (N)")
plt.legend()
plt.title("Hydrodynamic forces (L+D) in body frame")
plt.savefig(dir+prefix+"hydro_forces.png")

plt.figure()
plt.plot(angle_of_attack_degree.cpu(), norm_hydro_forces.cpu(), c='cyan', label="norm hydro forces")
plt.xlabel("angle of attack (degree)")
plt.ylabel("norm forces (N)")
plt.legend()
plt.title("Norm of hydrodynamic forces")
plt.savefig(dir+prefix+"norm_hydro_forces.png")

plt.figure()
plt.plot(angle_of_attack_degree.cpu(), magn_lift.cpu(), '--', c='g', label="Lift")
plt.plot(angle_of_attack_degree.cpu(), magn_drag.cpu(), '--', c='y', label="Drag")
plt.xlabel("angle of attack (degree)")
plt.ylabel("forces (N)")
plt.legend()
plt.title("Magnitude of lift and drag")
plt.savefig(dir+prefix+"magn_lift_drag_aoa.png")

print(f"FIGURE SAVED IN {(dir+prefix).upper()}")