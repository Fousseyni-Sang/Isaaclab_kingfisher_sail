# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.physics.aerodynamics import AerodynamicsCfg, Aerodynamics
import torch
import matplotlib.pyplot as plt
#from .kingfisher_sail_env import KingfisherSailEnv, KingfisherSailEnvCfg
import os
import pandas as pd
# Aerodynamics
aerodynamics_cfg: AerodynamicsCfg = AerodynamicsCfg()
aerodynamics_cfg.air_density = 1.225
aerodynamics_cfg.wing_span = 1
aerodynamics_cfg.wing_chord = 0.2
aerodynamics_cfg.wind_direction = -90*torch.pi/180
aerodynamics_cfg.wind_speed = 5
aerodynamics_cfg.angle_of_attack = 20*torch.pi/180
device = 'cuda:0'


min_aoa = -180
max_aoa = 180
num_envs = int(abs(min_aoa) + abs(max_aoa))
aerodynamics = Aerodynamics(num_envs=num_envs, device=device, cfg=aerodynamics_cfg, naca_profile="0012")

angle_of_attack_degree = torch.linspace(min_aoa, max_aoa, num_envs, device=device)
angle_of_attack_radian = (torch.pi/180)*angle_of_attack_degree
#cl, cd = aerodynamics.generate_coeffs(angle_of_attack_radian)

dir = "/mnt/gpu_storage/zrr/fsangare/asv-sawasp-fousseyni/Isaaclab_kingfisher_sail/Test_plot/sail_aerodynamics/"
prefix = "sail_"
os.makedirs(dir, exist_ok=True)

ship_heading_w = (torch.pi/180)*0.0*torch.ones_like(angle_of_attack_radian)  # ship heading in world
ship_lin_vel2D = torch.tensor([[0.0, 0.0]], device=device).repeat(angle_of_attack_radian.shape[0], 1)  # ship linear velocity in 2D
true_wind_vel2D = aerodynamics.generate_true_wind_components(aerodynamics.Uw, aerodynamics.Beta_w, ship_heading_w)
apparent_wind_vel2D = aerodynamics.generate_apparent_wind_components(true_wind_vel2D, ship_lin_vel2D)
apparent_wind_angle = aerodynamics.get_apparent_wind_angle(apparent_wind_vel2D)
foil_angl = aerodynamics.get_sail_angle(apparent_wind_angle, angle_of_attack_radian)

hydro_forces = aerodynamics.compute_wind_effect(aerodynamics.Uw, aerodynamics.Beta_w, ship_heading_w, ship_lin_vel2D, angle_of_attack_radian, foil_angl)
norm_hydro_forces = torch.norm(hydro_forces, dim=1)

magn_lift, magn_drag = aerodynamics.lift_magnitude, aerodynamics.drag_magnitude
vec_lift, vec_drag = aerodynamics.wind_lift_b, aerodynamics.wind_drag_b

cl = aerodynamics.lift_coeff
cd = aerodynamics.drag_coeff

name = ['angle_of_attack_degree', 'cl', 'cd', 'lift_magnitude', 'drag_magnitude']
data = torch.stack([angle_of_attack_degree.cpu(), cl.cpu(), cd.cpu(), magn_lift.cpu(), magn_drag.cpu()], dim=1)
df = pd.DataFrame(data.numpy(), columns=name)
df.to_csv(dir+prefix+"sail_aerodynamics_data.csv", index=False)

# Open Aerodynamics figure:  xdg-open /tmp/cl_cd_aoa.png
plt.figure()
plt.plot(angle_of_attack_degree.cpu(), cl.cpu(), '--', c='g', label="cl")
plt.plot(angle_of_attack_degree.cpu(), cd.cpu(), '--', c='y', label="cd")
plt.xlabel("angle of attack (degree)")
plt.ylabel("cl and cd")
plt.legend()
plt.savefig(dir+prefix+"cl_cd_aoa.png")

plt.figure()
plt.plot(angle_of_attack_degree.cpu(), (cl/cd).cpu(), c='g', label="cl/cd")
plt.xlabel("angle of attack (degree)")
plt.ylabel("cl/cd")
plt.legend()
plt.savefig(dir+prefix+"cl_cd_ratio.png")

plt.figure()
plt.plot(cd.cpu(), cl.cpu(), c='grey', label="cl vs cd")
plt.xlabel("cd")
plt.ylabel("cl")
plt.legend()
plt.savefig(dir+prefix+"cl_vs_cd.png")

plt.figure()
plt.plot(angle_of_attack_degree.cpu(), vec_lift.cpu(), '--', c='g', label="Lift")
plt.plot(angle_of_attack_degree.cpu(), vec_drag.cpu(), '--', c='y', label="Drag")
plt.xlabel("angle of attack (degree)")
plt.ylabel("forces (N)")
plt.title("Vector components of lift and drag")
plt.legend()
plt.savefig(dir+prefix+"vec_lift_drag_aoa.png")

plt.figure()
plt.plot(angle_of_attack_degree.cpu(), magn_lift.cpu(), '--', c='g', label="Lift")
plt.plot(angle_of_attack_degree.cpu(), magn_drag.cpu(), '--', c='y', label="Drag")
plt.xlabel("angle of attack (degree)")
plt.ylabel("forces (N)")
plt.legend()
plt.title("Magnitude of lift and drag")
plt.savefig(dir+prefix+"magn_lift_drag_aoa.png")

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