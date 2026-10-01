
"""
Brute-force sailing polar evaluation.

For every True Wind Speed (TWS) and True Wind Angle (TWA):

    - Create 360 parallel environments.
    - Each environment receives one fixed sail-angle command.
    - Run the episode for the full episode duration.
    - Record maximum boat speed and maximum VMG.
    - Select the sail angle producing the maximum boat speed.
    - Save the result to CSV.

The 360 sail commands are:

    linspace(-1, 1, 360)

which correspond to:

    -180 deg ... +180 deg

assuming the environment maps action [-1, 1] -> sail angle [-pi, pi].
"""

import argparse
import os
import math
import csv

import torch
import numpy as np

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Brute-force sailing polar evaluation."
)

parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Registered Isaac Lab polar environment.",
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=360,
    help="Number of parallel sail-angle trials.",
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
)

parser.add_argument(
    "--disable_fabric",
    action="store_true",
)

parser.add_argument(
    "--output",
    type=str,
    default="polar_data.csv",
)

# Optional checkpoint arguments are not needed because this is
# a direct physics sweep, not RL policy evaluation.

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Isaac Lab imports
# -----------------------------------------------------------------------------

import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg


# -----------------------------------------------------------------------------
# Polar configuration
# -----------------------------------------------------------------------------

NUM_SAIL_ANGLES = 360

SAIL_ANGLES_DEG = np.linspace(
    -180.0,
    180.0,
    NUM_SAIL_ANGLES,
)

# Action convention:
#
#   -1 -> -pi
#    0 ->  0
#   +1 -> +pi
#
SAIL_ACTIONS = torch.linspace(
    -1.0,
    1.0,
    NUM_SAIL_ANGLES,
)

TRUE_WIND_ANGLES_DEG = np.arange(
    0.0,
    181.0,
    5.0,
)

TRUE_WIND_SPEEDS_MS = np.arange(
    1.0,
    6.0,
    1.0,
)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():

    # -------------------------------------------------------------------------
    # Environment configuration
    # -------------------------------------------------------------------------

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=NUM_SAIL_ANGLES,
        use_fabric=not args_cli.disable_fabric,
    )

    # Make absolutely sure we have 360 environments.
    env_cfg.scene.num_envs = NUM_SAIL_ANGLES

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
    )

    env = env.unwrapped

    print()
    print("=" * 80)
    print("BRUTE FORCE SAILING POLAR")
    print("=" * 80)
    print(f"Parallel environments : {NUM_SAIL_ANGLES}")
    print(f"Sail angles           : {SAIL_ANGLES_DEG[0]:.2f} ... "
          f"{SAIL_ANGLES_DEG[-1]:.2f} deg")
    print(f"TWA                   : {TRUE_WIND_ANGLES_DEG}")
    print(f"TWS                   : {TRUE_WIND_SPEEDS_MS}")
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    os.makedirs(
        os.path.dirname(args_cli.output)
        if os.path.dirname(args_cli.output)
        else ".",
        exist_ok=True,
    )

    results = []

    # -------------------------------------------------------------------------
    # Fixed sail actions
    # -------------------------------------------------------------------------

    sail_actions = SAIL_ACTIONS.to(
        device=env.device,
        dtype=torch.float32,
    )

    # Shape expected by the environment.
    #
    # If your environment has 3 actions:
    #
    #   [left_thruster, right_thruster, sail]
    #
    # then we construct:
    #
    #   [0, 0, sail]
    #
    # If it only has one action, change this section accordingly.

    action_dim = env.action_space.shape[0]

    actions = torch.zeros(
        NUM_SAIL_ANGLES,
        action_dim,
        device=env.device,
    )

    actions[:, -1] = sail_actions

    # -------------------------------------------------------------------------
    # Loop over wind speeds
    # -------------------------------------------------------------------------

    for tws in TRUE_WIND_SPEEDS_MS:

        print()
        print("#" * 80)
        print(f"TRUE WIND SPEED = {tws:.1f} m/s")
        print("#" * 80)

        # ---------------------------------------------------------------------
        # Loop over wind angles
        # ---------------------------------------------------------------------

        for twa_deg in TRUE_WIND_ANGLES_DEG:

            print()
            print(
                f"[POLAR] TWS = {tws:.1f} m/s   "
                f"TWA = {twa_deg:.1f} deg"
            )

            # -----------------------------------------------------------------
            # Reset ALL 360 trials
            # -----------------------------------------------------------------

            obs, _ = env.reset()

            # -----------------------------------------------------------------
            # Set wind direction
            # -----------------------------------------------------------------

            wind_angle_rad = torch.tensor(
                np.deg2rad(twa_deg),
                device=env.device,
                dtype=torch.float32,
            )

            env._sail_aerodynamics.update_flow(
                flow_direction=wind_angle_rad
            )

            # -----------------------------------------------------------------
            # Set wind speed
            # -----------------------------------------------------------------
            #
            # IMPORTANT:
            #
            # Your existing code exposes the wind speed internally as
            # dynamics.Uw (you already referenced it in your debug code).
            #
            # If Uw is a tensor, this works:
            #
            #     env._sail_aerodynamics.Uw[:] = tws
            #
            # If your actual wind-speed variable lives somewhere else,
            # replace ONLY this line.
            # -----------------------------------------------------------------

            env._sail_aerodynamics.Uw[:] = tws

            # Keep actuator dynamics synchronized if it uses the same
            # FoilDynamics object.
            env._sail_actuator.dynamics.Uw[:] = tws

            # -----------------------------------------------------------------
            # Per-sail-angle metrics
            # -----------------------------------------------------------------

            max_speed = torch.full(
                (NUM_SAIL_ANGLES,),
                -float("inf"),
                device=env.device,
            )

            max_vmg = torch.full(
                (NUM_SAIL_ANGLES,),
                -float("inf"),
                device=env.device,
            )

            final_speed = torch.zeros(
                NUM_SAIL_ANGLES,
                device=env.device,
            )

            final_vmg = torch.zeros(
                NUM_SAIL_ANGLES,
                device=env.device,
            )

            max_force_forward = torch.full(
                (NUM_SAIL_ANGLES,),
                -float("inf"),
                device=env.device,
            )

            # -----------------------------------------------------------------
            # Run episode
            # -----------------------------------------------------------------

            done = torch.zeros(
                NUM_SAIL_ANGLES,
                dtype=torch.bool,
                device=env.device,
            )

            step = 0

            while simulation_app.is_running():

                with torch.inference_mode():

                    # ---------------------------------------------------------
                    # Apply fixed sail-angle commands
                    # ---------------------------------------------------------

                    obs, reward, terminated, truncated, extras = env.step(
                        actions
                    )

                    # ---------------------------------------------------------
                    # Boat velocity in body frame
                    # ---------------------------------------------------------

                    velocity_b = env._robot.data.root_lin_vel_b

                    # Forward boat speed.
                    #
                    # x-axis is the boat longitudinal direction.
                    speed = velocity_b[:, 0]

                    # Absolute horizontal speed can also be useful.
                    speed_norm = torch.linalg.norm(
                        velocity_b[:, :2],
                        dim=-1,
                    )

                    # ---------------------------------------------------------
                    # VMG relative to the prescribed TWA
                    # ---------------------------------------------------------
                    #
                    # For a fixed boat heading:
                    #
                    #     VMG = V * cos(TWA)
                    #
                    # This is the conventional signed VMG:
                    #
                    #   TWA < 90  -> positive upwind progress
                    #   TWA = 90  -> zero
                    #   TWA > 90  -> negative if interpreted as upwind VMG
                    #
                    # We use forward boat speed here.
                    # ---------------------------------------------------------

                    vmg = speed * np.cos(
                        np.deg2rad(twa_deg)
                    )

                    # ---------------------------------------------------------
                    # Update maximum speed
                    # ---------------------------------------------------------

                    max_speed = torch.maximum(
                        max_speed,
                        speed,
                    )

                    max_vmg = torch.maximum(
                        max_vmg,
                        vmg,
                    )

                    final_speed = speed
                    final_vmg = vmg

                    # ---------------------------------------------------------
                    # Forward aerodynamic force
                    # ---------------------------------------------------------

                    sail_force_b = (
                        env._sail_aerodynamic_force_b[:, 0, :3]
                    )

                    force_forward = sail_force_b[:, 0]

                    max_force_forward = torch.maximum(
                        max_force_forward,
                        force_forward,
                    )

                    # ---------------------------------------------------------
                    # Episode termination
                    # ---------------------------------------------------------

                    done = torch.logical_or(
                        terminated,
                        truncated,
                    )

                    step += 1

                    if torch.all(done):
                        break

            # -----------------------------------------------------------------
            # Select best sail angle
            # -----------------------------------------------------------------

            best_idx = torch.argmax(max_speed)

            best_sail_angle_deg = (
                SAIL_ANGLES_DEG[best_idx.item()]
            )

            best_sail_action = (
                sail_actions[best_idx].item()
            )

            best_speed = (
                max_speed[best_idx].item()
            )

            best_vmg = (
                max_vmg[best_idx].item()
            )

            best_final_speed = (
                final_speed[best_idx].item()
            )

            best_final_vmg = (
                final_vmg[best_idx].item()
            )

            best_force_forward = (
                max_force_forward[best_idx].item()
            )

            print(
                f"    best sail angle : {best_sail_angle_deg:8.2f} deg"
            )
            print(
                f"    action          : {best_sail_action:8.4f}"
            )
            print(
                f"    max speed       : {best_speed:8.4f} m/s"
            )
            print(
                f"    max VMG         : {best_vmg:8.4f} m/s"
            )
            print(
                f"    final speed     : {best_final_speed:8.4f} m/s"
            )
            print(
                f"    final VMG       : {best_final_vmg:8.4f} m/s"
            )
            print(
                f"    max F_forward   : {best_force_forward:8.4f} N"
            )

            # -----------------------------------------------------------------
            # Save result
            # -----------------------------------------------------------------

            results.append(
                {
                    "wind_angle_deg": float(twa_deg),
                    "wind_speed_ms": float(tws),

                    "optimal_sail_angle_deg":
                        float(best_sail_angle_deg),

                    "optimal_sail_action":
                        float(best_sail_action),

                    "max_boat_speed_ms":
                        float(best_speed),

                    "max_vmg_ms":
                        float(best_vmg),

                    "final_boat_speed_ms":
                        float(best_final_speed),

                    "final_vmg_ms":
                        float(best_final_vmg),

                    "max_forward_sail_force_N":
                        float(best_force_forward),

                    "num_sail_angles":
                        NUM_SAIL_ANGLES,

                    "episode_steps":
                        step,
                }
            )

            # -----------------------------------------------------------------
            # Write CSV after EVERY condition
            #
            # This is deliberate: if Isaac crashes at TWA=135 deg, you don't
            # lose everything from 0 ... 130 deg.
            # -----------------------------------------------------------------

            with open(
                args_cli.output,
                "w",
                newline="",
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=results[0].keys(),
                )

                writer.writeheader()
                writer.writerows(results)

    # -------------------------------------------------------------------------
    # Finished
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("POLAR EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Saved: {args_cli.output}")
    print(f"Rows : {len(results)}")
    print("=" * 80)

    env.close()


if __name__ == "__main__":

    try:
        main()

    finally:
        simulation_app.close()

