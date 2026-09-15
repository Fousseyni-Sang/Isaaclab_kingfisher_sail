"""
Evaluation sweep for the classical clause-hauled tacking + hybrid-thruster
controller (classical_controller.py). Structurally this mirrors
sweep_eval_loop.py (the RL policy's sweep) almost exactly -- same per-env
completion tracking, same reset-boundary-row fix, same goal_reached/timed_out
sourcing from the sim -- the only real difference is WHAT computes the
action each step (a hand-coded control law instead of agent.get_action()).

Integrate this into your play/eval script the same way sweep_eval_loop.py
was integrated -- after env creation, replacing the `while ... episodes_done`
block. No trained agent/checkpoint needed at all for this script.

Output CSV intentionally reuses the RL sweep's exact column names
(`energy_context` instead of `assist_level`, etc.) so it can be fed straight
into episodes_from_timeseries()/plot_pareto_front() from pareto_energy_time.py
with zero changes -- only an added `method` column distinguishes it.
"""

import numpy as np
import torch
import pandas as pd

from classical_controller import (
    ClassicalControllerState, classical_action, PolarTable, compute_max_ld_angle_deg,
)

NUM_ANGLES = 5
NUM_SPEEDS = 9
EPISODES_PER_ENV = NUM_ANGLES * NUM_SPEEDS  # 45 -- one full local wind sweep

# The classical analogue of the RL policy's `energy_context` sweep. Chosen to
# span the same [0.1, 1.1]-ish range for a fair visual comparison; adjust
# freely -- more points give a smoother classical curve at the cost of
# EPISODES_PER_ENV x len(ASSIST_LEVELS) total episodes.
ASSIST_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

# Path to the (wind_speed, wind_angle_deg, boat_speed) CSV behind your
# sailing_polar_max_vmg plot, if you have it. None falls back to a fixed
# close-hauled angle (see PolarTable/FALLBACK_UPWIND_ANGLE_DEG in
# classical_controller.py).
POLAR_CSV_PATH = "sail_sweep_results_1_copy.csv"  # your real polar sweep data

num_envs = env.unwrapped.num_envs
device = env.unwrapped.device
env.unwrapped.is_Training = False
env.unwrapped.is_paper_eval = True

polar_table = PolarTable(POLAR_CSV_PATH)
max_ld_angle_deg = compute_max_ld_angle_deg(env.unwrapped._sail_aerodynamics, device)

all_rows = []

for assist_level in ASSIST_LEVELS:
    print(f"[INFO] Classical baseline sweep: assist_level={assist_level} "
          f"({EPISODES_PER_ENV} episodes/env x {num_envs} envs)")

    # Restart this assist level's wind cycle from (angle_idx=0, speed_idx=0).
    # A plain env.reset() does NOT zero these -- they only advance on their
    # own via _reset_idx's is_paper_eval branch -- so we zero them explicitly
    # to get a clean, comparable 45-episode sweep for every assist_level.
    env.unwrapped.paper_wind_angle_idx.zero_()
    env.unwrapped.paper_wind_speed_idx.zero_()

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    info = env.unwrapped.extras.get("info", {})

    state = ClassicalControllerState(num_envs, device=device)
    state.reset(torch.arange(num_envs, device=device), info)

    episodes_done_per_env = np.zeros(num_envs, dtype=int)
    per_env_buffers = [[] for _ in range(num_envs)]

    while not np.all(episodes_done_per_env >= EPISODES_PER_ENV):
        with torch.inference_mode():
            info = env.unwrapped.extras.get("info", {})
            actions = classical_action(info, state, assist_level, polar_table, max_ld_angle_deg)
            obs, rew, dones, extras = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]

            info = extras.get("info", {})
            dones_np = dones.detach().cpu().numpy().astype(bool)

            for env_id in range(num_envs):
                if episodes_done_per_env[env_id] >= EPISODES_PER_ENV:
                    continue

                row = dict(
                    env_id=env_id,
                    time_step=len(per_env_buffers[env_id]),
                    energy=info["energy"][env_id].item(),
                    distance=info["distance"][env_id].item(),
                    energy_context=assist_level,   # renamed for pipeline compatibility
                    true_wind_speed=info["true_wind_speed"][env_id].item(),
                    true_wind_angle_w=info["true_wind_angle_w"][env_id].item(),
                )

                if dones_np[env_id]:
                    # Same reset-boundary contamination as the RL sweep: this
                    # row is already the NEXT episode's post-reset state.
                    goal_reached = bool(env.unwrapped.reset_terminated[env_id].item())
                    timed_out = bool(env.unwrapped.reset_time_outs[env_id].item())

                    finished_rows = per_env_buffers[env_id]
                    local_ep_idx = episodes_done_per_env[env_id]
                    for r in finished_rows:
                        r["local_episode_idx"] = local_ep_idx
                        r["goal_reached"] = goal_reached
                        r["timed_out"] = timed_out
                    all_rows.extend(finished_rows)

                    episodes_done_per_env[env_id] += 1

                    # This env just reset internally -- reseed the
                    # controller's per-env state (new start_pos, fresh tack)
                    # and start its new episode buffer with this row as t=0.
                    state.reset(torch.tensor([env_id], device=device), info)
                    row["time_step"] = 0
                    per_env_buffers[env_id] = [row]
                else:
                    per_env_buffers[env_id].append(row)

    print(f"[INFO] assist_level={assist_level} done: all {num_envs} envs "
          f"completed {EPISODES_PER_ENV} episodes each.")

df = pd.DataFrame(all_rows)
df["method"] = "classical"
df.to_csv("classical_baseline_swept.csv", index=False)
print(f"[INFO] Saved {len(df)} rows across {len(ASSIST_LEVELS)} assist levels "
      f"x {EPISODES_PER_ENV} episodes x {num_envs} envs.")
