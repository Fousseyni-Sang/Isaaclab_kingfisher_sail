# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.physics.aerodynamics import AerodynamicsCfg, Aerodynamics
import torch
import matplotlib.pyplot as plt

# Aerodynamics
aerodynamics_cfg: AerodynamicsCfg = AerodynamicsCfg()
aerodynamics_cfg.air_density = 1.225
aerodynamics_cfg.wing_span = 1
aerodynamics_cfg.wing_chord = 0.2
aerodynamics_cfg.wind_direction = -90*torch.pi/180
aerodynamics_cfg.wind_speed = 5
aerodynamics_cfg.angle_of_attack = 20*torch.pi/180
device = 'cuda:0'

aerodynamics = Aerodynamics(num_envs=1, device=device, cfg=aerodynamics_cfg)

angle_of_attack = torch.linspace(-180, 180, 360, device=device)
cl, cd = aerodynamics.generate_coeffs(angle_of_attack)

"""plt.figure()
plt.plot(angle_of_attack.cpu(), cl.cpu(), '--', c='g', label="cl")
plt.plot(angle_of_attack.cpu(), cd.cpu(), '--', c='y', label="cd")
plt.xlabel("angle of attack (degree)")
plt.ylabel("cl and cd")
plt.legend()
plt.savefig("/tmp/cl_cd_aoa.png")

plt.figure()
plt.plot(angle_of_attack.cpu(), (cl/cd).cpu(), c='g', label="cl/cd")
plt.xlabel("angle of attack (degree)")
plt.ylabel("cl/cd")
plt.legend()
plt.savefig("/tmp/cl_cd_ratio.png")

plt.figure()
plt.plot(cd.cpu(), cl.cpu(), c='grey', label="cl vs cd")
plt.xlabel("cd")
plt.ylabel("cl")
plt.legend()
plt.savefig("/tmp/cl_vs_cd.png")"""

