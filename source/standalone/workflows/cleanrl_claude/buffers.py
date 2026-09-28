"""GPU replay buffer for vectorized envs, with optional HER and sequence sampling (for DRQN).

Data is stored time-major, [capacity_t, num_envs, ...], so one env's trajectory stays contiguous in time.
A "logical index" 0..size-1 means oldest..newest stored step; _phys() turns it into the row of the ring buffer.

HER here is NOT the SB3 design. SB3 asks the env for `compute_reward(achieved_goal, desired_goal, info)` and
works on dict observations. This buffer has no such callback: the goal-reached rule, the goal features of the
observation and the sparse reward are hard-coded in ReplayBuffer._relabel() for the Kingfisher env. What it expects:

  * the env exposes the robot pose (xy, yaw) and goal (xy) each step -> IsaacEnv.her_state(), stored by add();
  * three observation slots (HerCfg.goal_obs_idx) hold the goal in the robot frame:
    [cos(bearing), sin(bearing), distance / max_dist]; they are recomputed for a relabeled goal;
  * reward = HerCfg.goal_reward if the robot is within HerCfg.thr of the goal after the step, else 0, and that same
    event ends the episode. The env's own (dense) reward is ignored when HER is on;
  * strategy "future": the new goal is where the robot really was 1..k steps later in the same episode.
To use another task or reward rule, change her_state() in common.py and the marked lines in _relabel().
"""
from dataclasses import dataclass

import torch


@dataclass
class HerCfg:
    thr: float  # goal-reached distance (from the env cfg)
    max_dist: float  # distance normalisation used inside the observation
    k: int = 4  # a future goal is drawn from the next k steps of the same episode ("future" strategy)
    ratio: float = 0.8  # fraction of sampled transitions that get a relabeled goal
    goal_reward: float = 1.0  # sparse reward when the (possibly relabeled) goal is reached
    goal_obs_idx: tuple = (6, 7, 8)  # obs slots holding [cos(bearing), sin(bearing), distance / max_dist]


class ReplayBuffer:
    def __init__(self, capacity, num_envs, obs_dim, act_shape, device, act_dtype=torch.float32, her=None):
        self.T, self.N, self.device, self.her = max(capacity // num_envs, 2), num_envs, device, her
        z = lambda *s, dtype=torch.float32: torch.zeros(self.T, num_envs, *s, dtype=dtype, device=device)
        self.obs, self.next_obs, self.act = z(obs_dim), z(obs_dim), z(*act_shape, dtype=act_dtype)
        self.rew, self.term, self.done = z(), z(), z()  # term = terminated (no bootstrap), done = terminated or truncated
        self.ptr = self.size = 0  # ptr = next row to write, size = number of valid rows
        if her:  # extra per-step data to rebuild the goal features and the sparse reward
            self.pos, self.yaw, self.goal = z(2), z(), z(2)  # robot xy / yaw and goal xy BEFORE the step
            self.ep = z(dtype=torch.long)  # episode number of each step, per env (to stay inside one episode)
            self.ep_counter = torch.zeros(num_envs, dtype=torch.long, device=device)

    def add(self, obs, act, rew, term, trunc, next_obs, state=None):
        """Store one vectorized step (one transition per env). `state` = env.her_state() taken before the step."""
        p = self.ptr
        self.obs[p], self.act[p], self.rew[p], self.next_obs[p] = obs, act, rew, next_obs
        self.term[p], self.done[p] = term.float(), (term | trunc).float()
        if self.her:
            self.pos[p], self.yaw[p], self.goal[p] = state["pos"], state["yaw"], state["goal"]
            self.ep[p] = self.ep_counter
            self.ep_counter += (term | trunc).long()  # the next step of a finished env belongs to a new episode
        self.ptr, self.size = (p + 1) % self.T, min(self.size + 1, self.T)

    def _phys(self, i):
        """Logical index (0 = oldest stored step) -> physical row of the ring buffer."""
        return ((self.ptr - self.size) + i) % self.T

    def __len__(self):  # number of stored transitions (all envs)
        return self.size * self.N

    # ---------------------------------------------------------------- transitions
    def sample(self, batch):
        # with HER, the step after a sampled one must exist too (its pose is the "next pose"), hence size - 1
        i = torch.randint(0, self.size - 1 if self.her else self.size, (batch,), device=self.device)
        n = torch.randint(0, self.N, (batch,), device=self.device)
        t = self._phys(i)
        out = dict(obs=self.obs[t, n], act=self.act[t, n], rew=self.rew[t, n], term=self.term[t, n], next_obs=self.next_obs[t, n])
        return self._relabel(out, i, n, t) if self.her else out

    def _relabel(self, out, i, n, t):
        """Replace the goal of some sampled transitions (i = logical step, n = env, t = physical row)."""
        c, B = self.her, len(i)

        # 1. Pick a future step j in 1..k steps after i (clamped to the newest step). Its robot position is the new goal.
        #    Relabel only if j is in the same episode as i, and only for a random fraction `ratio` of the batch.
        j = (i + torch.randint(1, c.k + 1, (B,), device=self.device)).clamp(max=self.size - 1)
        tj, t1 = self._phys(j), self._phys(i + 1)  # t1: the step right after i, its stored pose is the pose AFTER step i
        relabel = (self.ep[tj, n] == self.ep[t, n]) & (torch.rand(B, device=self.device) < c.ratio)
        goal = torch.where(relabel[:, None], self.pos[tj, n], self.goal[t, n])  # (B, 2) goal used for each sample

        # 2. Rebuild the goal features of the observation in the robot frame, at time t (obs) and t+1 (next_obs):
        #    rotate the world vector robot->goal by -yaw, then [cos(bearing), sin(bearing), distance / max_dist].
        #    Only relabeled samples take the new features; the others keep the env's own observation.
        idx = list(c.goal_obs_idx)
        for key, tt in (("obs", t), ("next_obs", t1)):
            rel = goal - self.pos[tt, n]
            cs, sn = torch.cos(self.yaw[tt, n]), torch.sin(self.yaw[tt, n])
            x, y = cs * rel[:, 0] + sn * rel[:, 1], -sn * rel[:, 0] + cs * rel[:, 1]  # goal in the robot frame
            d = torch.hypot(x, y).clamp(min=1e-6)
            feats = torch.stack([x / d, y / d, d / c.max_dist], dim=-1)  # cos, sin of the bearing, normalised distance
            out[key][:, idx] = torch.where(relabel[:, None], feats, out[key][:, idx])

        # 3. Sparse reward and termination. Relabeled: reached = robot after step i is within thr of the new goal
        #    (the pose after step i is only valid inside the same episode, which `relabel` guarantees).
        #    Not relabeled: reached = the env's own `terminated` flag (it means the real goal was reached).
        reached_new = torch.linalg.norm(self.pos[t1, n] - goal, dim=-1) < c.thr
        reached = torch.where(relabel, reached_new, out["term"] > 0)
        out["rew"], out["term"] = reached.float() * c.goal_reward, reached.float()
        return out

    # ---------------------------------------------------------------- sequences
    def sample_sequences(self, batch, length):
        """Random windows of consecutive steps of one env.

        Returns batch-first tensors: obs (B, L+1, D) (one extra obs so the target of the last step can be computed);
        act, rew, term, done (B, L). Windows may cross episode ends; `done` tells the recurrent net where to reset.
        """
        i = torch.randint(0, self.size - length, (batch,), device=self.device)  # start of each window (logical)
        n = torch.randint(0, self.N, (batch,), device=self.device)
        t = self._phys(i[:, None] + torch.arange(length + 1, device=self.device))  # (B, L+1) physical rows
        env = n[:, None]
        return dict(obs=self.obs[t, env], act=self.act[t[:, :-1], env], rew=self.rew[t[:, :-1], env],
                    term=self.term[t[:, :-1], env], done=self.done[t[:, :-1], env])
