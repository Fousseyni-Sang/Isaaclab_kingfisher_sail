"""
Corrected replacement for the `while ... episodes_finished < num_episode:`
block in play_checkpoint.py.

Key realization from reading KingfisherSailEnv._reset_idx: the env ALREADY
sweeps wind conditions per-env, deterministically, via paper_wind_angle_idx /
paper_wind_speed_idx (each shape (num_envs,)), advancing independently every
time that specific env resets. energy_context per env is fixed
(paper_eval_context = linspace(0.1, 1.1, num_envs), re-applied unchanged on
every reset). So there's no need to synchronize resets across envs -- you
only need to:

  (a) stop once EVERY env has completed a full local cycle (5 angles x 9
      speeds = 45 episodes each), not once a GLOBAL counter hits a target.
  (b) fix a real off-by-one in the original logging: the row logged at the
      exact step where an env's `done` fires already contains the NEXT
      episode's post-reset info (Isaac Lab resets inside env.step() before
      get_info() runs). The original script appended it to the OLD
      episode's buffer and then discarded it when clearing that buffer --
      so the old episode got a mislabeled trailing row, and the new episode
      silently lost its own true first row. Fixed below by routing that row
      to the NEW episode's buffer instead.

RECOMMENDED one-line addition to KingfisherSailEnv.get_info() (not required,
but removes all guesswork about which (angle,speed) index produced a given
row -- add to the returned dict):

    "wind_angle_idx": self.paper_wind_angle_idx,   # (N,)
    "wind_speed_idx": self.paper_wind_speed_idx,   # (N,)

With that in place, group your final CSV by (env_id, wind_angle_idx,
wind_speed_idx) instead of inferring the condition from float wind values.
"""

import numpy as np
import torch
import pandas as pd

NUM_ANGLES = 5   # len(paper_wind_angle) in the env
NUM_SPEEDS = 9   # len(paper_wind_speed) in the env
EPISODES_PER_ENV = NUM_ANGLES * NUM_SPEEDS  # 45 -- one full local sweep

num_envs = env.unwrapped.num_envs
env.unwrapped.is_Training = False
env.unwrapped.is_paper_eval = True

obs = env.reset()
if isinstance(obs, dict):
    obs = obs["obs"]
if agent.is_rnn:
    agent.init_rnn()

# Per-env bookkeeping.
episodes_done_per_env = np.zeros(num_envs, dtype=int)
per_env_buffers = [[] for _ in range(num_envs)]   # current in-progress episode, per env
all_rows = []                                      # finished rows, flat list of dicts
prev_dones = np.zeros(num_envs, dtype=bool)         # was this env done last iteration

print(f"[INFO] Target: {EPISODES_PER_ENV} episodes/env x {num_envs} envs "
      f"= {EPISODES_PER_ENV * num_envs} total episodes")

while not np.all(episodes_done_per_env >= EPISODES_PER_ENV):
    with torch.inference_mode():
        obs_t = agent.obs_to_torch(obs)
        actions = agent.get_action(obs_t, is_deterministic=agent.is_deterministic)
        obs, rew, dones, extras = env.step(actions)
        if isinstance(obs, dict):
            obs = obs["obs"]

        info = extras.get("info", {})
        dones_np = dones.detach().cpu().numpy().astype(bool)

        for env_id in range(num_envs):
            # Skip envs that have already completed their full local sweep --
            # they'll keep auto-resetting/stepping under the hood but we no
            # longer care about their data.
            if episodes_done_per_env[env_id] >= EPISODES_PER_ENV:
                continue

            row = dict(
                env_id=env_id,
                time_step=len(per_env_buffers[env_id]),
                energy=info.get("energy", torch.zeros(num_envs))[env_id].item(),
                distance=info.get("distance", torch.zeros(num_envs))[env_id].item(),
                energy_context=info.get("energy_context", torch.zeros(num_envs))[env_id].item(),
                true_wind_speed=info.get("true_wind_speed", torch.zeros(num_envs))[env_id].item(),
                true_wind_angle_w=info.get("true_wind_angle_w", torch.zeros(num_envs))[env_id].item(),
                # If you add the recommended get_info() fields, uncomment:
                # wind_angle_idx=info.get("wind_angle_idx", torch.zeros(num_envs))[env_id].item(),
                # wind_speed_idx=info.get("wind_speed_idx", torch.zeros(num_envs))[env_id].item(),
                # ... add every other field you currently log ...
            )

            if dones_np[env_id]:
                # This row is contaminated -- it's already the NEXT episode's
                # post-reset state, not the finishing episode's true last
                # step (see module docstring). Because of that, the OLD
                # episode's last KEPT row will always be "one step before"
                # success/timeout (e.g. distance ~0.30-0.31 when
                # goal_reached_threshold=0.3) -- never the crossing point
                # itself. Don't try to infer success from distance
                # thresholding on the kept rows; read the sim's own
                # termination reason instead, which is unambiguous.
                goal_reached = bool(env.unwrapped.reset_terminated[env_id].item())
                timed_out = bool(env.unwrapped.reset_time_outs[env_id].item())

                finished_episode_rows = per_env_buffers[env_id]
                local_ep_idx = episodes_done_per_env[env_id]
                for r in finished_episode_rows:
                    r["local_episode_idx"] = local_ep_idx
                    r["goal_reached"] = goal_reached
                    r["timed_out"] = timed_out
                all_rows.extend(finished_episode_rows)

                episodes_done_per_env[env_id] += 1

                row["time_step"] = 0
                per_env_buffers[env_id] = [row]
            else:
                per_env_buffers[env_id].append(row)

        if agent.is_rnn and agent.states is not None:
            for s in agent.states:
                s[:, dones, :] = 0.0

print("[INFO] All envs completed their full local wind sweep.")
df = pd.DataFrame(all_rows)
df.to_csv("all_steps_swept.csv", index=False)
print(f"[INFO] Saved {len(df)} rows, "
      f"{df.groupby(['env_id','local_episode_idx']).ngroups} episodes total.")
