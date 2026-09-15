# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.physics.aerodynamics import Aerodynamics, AerodynamicsCfg
import torch
import matplotlib.pyplot as plt
import numpy as np

# Aerodynamics
aerodynamics_cfg: AerodynamicsCfg = AerodynamicsCfg()
aerodynamics_cfg.air_density = 1.225
aerodynamics_cfg.wing_span = 1
aerodynamics_cfg.wing_chord = 0.2
aerodynamics_cfg.wind_direction = -90*torch.pi/180
aerodynamics_cfg.wind_speed = 8
aerodynamics_cfg.angle_of_attack = 20*torch.pi/180
aerodynamics_cfg.min_upwind_angle = 45*torch.pi/180
aerodynamics_cfg.max_downwind_angle = 140*torch.pi/180
device = 'cpu'

min_target_distance = 20.0
max_target_distance = 100.0
min_target_bearing = 0 #-torch.pi / 2
max_target_bearing = 5*torch.pi/180 #torch.pi / 2
max_cross_track = 8.0
num_envs = 1  # Number of environments

in_tack_mode = torch.ones(num_envs, device=device)

tack_length = torch.ones(num_envs, device=device)  # Default tack leg length
num_tack_waypoints = 10  # Default number of waypoints for tacking
tack_waypoints = torch.zeros((num_envs, num_tack_waypoints, 3), device=device)  # 10 waypoints
tack_valid_mask = torch.zeros((num_envs, num_tack_waypoints), dtype=torch.bool, device=device)

_desired_pos_w = torch.zeros(num_envs, 3, device=device)
initial_distance = torch.zeros_like(_desired_pos_w[0, 0]).uniform_(
            min_target_distance, max_target_distance
        )

initial_robot_pos = torch.zeros((num_envs, 3), device=device)  # Initial robot position
initial_bearing = torch.zeros_like(_desired_pos_w[0, 0]).uniform_(
            min_target_bearing, max_target_bearing
        )
distance = initial_distance
previous_distance = initial_distance
_desired_pos_w[0, 0] = torch.cos(initial_bearing) * initial_distance
_desired_pos_w[0, 1] = torch.sin(initial_bearing) * initial_distance
_desired_pos_w[0, 2] = 0.0  # only in 2D


aerodynamics = Aerodynamics(num_envs=1, device=device, cfg=aerodynamics_cfg)

dir = "/home/GTL/fsangare/Isaaclab_kingfisher_sail/"

angle_of_attack = torch.linspace(-180, 180, 360, device=device)
cl, cd = aerodynamics.generate_coeffs(angle_of_attack)

aoa_np = angle_of_attack.cpu().numpy()
cl_np = cl.cpu().numpy()
cd_np = cd.cpu().numpy()

# ---------------------------------------------------------------------------
# The XFOIL-fitted range is bounded by the STALL angles closest to alpha=0 --
# i.e. the first local maximum of CL encountered walking outward from 0
# toward +180 (positive stall), and the first local minimum walking outward
# from 0 toward -180 (negative stall).
#
# IMPORTANT: this must NOT be a global argmax/argmin over the full sweep.
# The Viterna post-stall model produces its own secondary lobes further out
# (typically past +/-90 deg) that are often as tall as, or taller than, the
# real near-zero stall peaks -- a global argmax/argmin picks those far
# Viterna-only extrema instead of the actual XFOIL stall points. Searching
# outward from alpha=0 and stopping at the FIRST local extremum avoids that.
# ---------------------------------------------------------------------------
def find_first_local_extremum(y: np.ndarray, start_idx: int, kind: str, direction: int) -> int:
    """Walk from start_idx in `direction` (+1 or -1) and return the index of
    the first local max (kind='max') or local min (kind='min') encountered.
    Falls back to start_idx if none is found before hitting an array edge."""
    n = len(y)
    i = start_idx
    while 0 < i < n - 1:
        if kind == 'max' and y[i - 1] < y[i] > y[i + 1]:
            return i
        if kind == 'min' and y[i - 1] > y[i] < y[i + 1]:
            return i
        i += direction
    return start_idx


idx0 = int(np.argmin(np.abs(aoa_np)))  # index of alpha closest to 0
idx_hi = find_first_local_extremum(cl_np, idx0, kind='max', direction=+1)  # positive stall (CL_max near 0)
idx_lo = find_first_local_extremum(cl_np, idx0, kind='min', direction=-1)  # negative stall (CL_min near 0)

alpha_stall_lo = aoa_np[idx_lo]
alpha_stall_hi = aoa_np[idx_hi]

print(f"[INFO] XFOIL-fitted range: alpha in [{alpha_stall_lo:.1f}, {alpha_stall_hi:.1f}] deg "
      f"(indices {idx_lo} to {idx_hi} out of {len(aoa_np)})")


def plot_xfoil_vs_extrapolated(ax, x, y, idx_lo, idx_hi, color_xfoil, color_extrap, label_prefix):
    """
    Plot y vs x with two visual regimes:
      - [idx_lo, idx_hi] (inclusive): the XFOIL-fitted stall-to-stall range,
        solid line, labeled "<label_prefix> (XFOIL)".
      - everything outside that range (both tails): the Viterna post-stall
        extrapolation, dashed line, a single shared legend entry
        "<label_prefix> (Viterna extrapolation)" even though it's drawn as
        two separate segments (left tail + right tail), to avoid a duplicate
        legend entry for what is physically the same regime.

    Each tail includes the boundary index itself so the dashed and solid
    segments touch with no visual gap at the stall angle.
    """
    extrap_label_used = False

    if idx_lo > 0:
        ax.plot(x[:idx_lo + 1], y[:idx_lo + 1], '--', color=color_extrap,
                 label=f"{label_prefix} (Viterna extrapolation)")
        extrap_label_used = True

    ax.plot(x[idx_lo:idx_hi + 1], y[idx_lo:idx_hi + 1], '-', color=color_xfoil,
             label=f"{label_prefix} (XFOIL)")

    if idx_hi < len(x) - 1:
        ax.plot(x[idx_hi:], y[idx_hi:], '--', color=color_extrap,
                 label=None if extrap_label_used else f"{label_prefix} (Viterna extrapolation)")

font_size=16
# --- Figure 1: CL and CD vs angle of attack, XFOIL range vs extrapolation ---
plt.figure() #figsize=(15, 6)
ax = plt.gca()
plot_xfoil_vs_extrapolated(ax, aoa_np, cl_np, idx_lo, idx_hi,
                            color_xfoil='g', color_extrap='limegreen', label_prefix="cl")
plot_xfoil_vs_extrapolated(ax, aoa_np, cd_np, idx_lo, idx_hi,
                            color_xfoil='goldenrod', color_extrap='khaki', label_prefix="cd")
ax.axvline(alpha_stall_lo, color='gray', linestyle=':', linewidth=1)
ax.axvline(alpha_stall_hi, color='gray', linestyle=':', linewidth=1)
plt.xlabel("angle of attack (degree)", fontsize=font_size)
plt.ylabel("cl and cd", fontsize=font_size)

plt.xticks(np.arange(-180, 180, 30))
plt.yticks(np.arange(-1.4, 1.4, 0.2)) 
plt.legend()
plt.savefig(dir+"cl_cd_aoa.pdf")

# --- Figure 2: CL/CD ratio, same XFOIL-vs-extrapolation split -------------
cl_cd_ratio = (cl / cd).cpu().numpy()
plt.figure()
ax2 = plt.gca()
plot_xfoil_vs_extrapolated(ax2, aoa_np, cl_cd_ratio, idx_lo, idx_hi,
                            color_xfoil='g', color_extrap='limegreen', label_prefix="cl/cd")
ax2.axvline(alpha_stall_lo, color='gray', linestyle=':', linewidth=1)
ax2.axvline(alpha_stall_hi, color='gray', linestyle=':', linewidth=1)
plt.xlabel("angle of attack (degree)")
plt.ylabel("cl/cd")
plt.legend()
plt.savefig(dir+"cl_cd_ratio.png")

# --- Figure 3: CL vs CD parametric plot, same split -----------------------
plt.figure()
ax3 = plt.gca()
if idx_lo > 0:
    ax3.plot(cd_np[:idx_lo + 1], cl_np[:idx_lo + 1], '--', color='lightgray',
              label="cl vs cd (Viterna extrapolation)")
ax3.plot(cd_np[idx_lo:idx_hi + 1], cl_np[idx_lo:idx_hi + 1], '-', color='dimgray',
          label="cl vs cd (XFOIL)")
if idx_hi < len(aoa_np) - 1:
    ax3.plot(cd_np[idx_hi:], cl_np[idx_hi:], '--', color='lightgray')
plt.xlabel("cd")
plt.ylabel("cl")
plt.legend()
plt.savefig(dir+"cl_vs_cd.png")