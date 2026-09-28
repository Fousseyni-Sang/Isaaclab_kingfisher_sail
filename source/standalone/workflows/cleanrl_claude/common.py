"""Shared helpers for the cleanrl_claude scripts.

Keeps every algorithm file short: CLI + Isaac Sim launch, a thin Isaac Lab env wrapper,
action discretization, logging, running statistics and GAE.
"""
import argparse
import dataclasses
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import tyro
import yaml


# ----------------------------------------------------------------------------- CLI
@dataclass
class BaseArgs:
    task: str = "Isaac-KingfisherSail-Direct-v0"
    num_envs: int = 64
    seed: int = 1
    total_timesteps: int = 0  # env steps summed over all envs (0 = algorithm default)
    exp_name: str = ""  # log sub-directory name (default: timestamp)
    log_root: str = "logs/cleanrl_claude"
    save_interval: int = 0  # save a checkpoint every N timesteps (0 = only at the end)
    result_file: str = ""  # json written at the end (used by tuner.py)
    cfg: str = ""  # yaml with default hyper-parameters (default: configs/<algo>.yaml); CLI flags win


@dataclass
class PPOArgs(BaseArgs):
    """PPO hyper-parameters. Defaults are overwritten by configs/<algo>.yaml (see load_cfg)."""

    num_envs: int = 4096
    max_epochs: int = 500  # PPO iterations, used when total_timesteps == 0
    num_steps: int = 48  # rollout length (rl_games: horizon_length)
    minibatch_size: int = 24576
    update_epochs: int = 5  # rl_games: mini_epochs
    gamma: float = 0.99
    gae_lambda: float = 0.95  # rl_games: tau
    learning_rate: float = 5e-4
    lr_schedule: str = "adaptive"  # adaptive | linear | constant
    kl_threshold: float = 0.016
    clip_coef: float = 0.2  # rl_games: e_clip
    ent_coef: float = 0.0
    vf_coef: float = 2.0  # rl_games: critic_coef
    max_grad_norm: float = 1.0
    bounds_loss_coef: float = 1e-4
    reward_scale: float = 0.01
    norm_adv: bool = True
    normalize_value: bool = True
    clip_vloss: bool = True
    normalize_input: bool = False
    clip_actions: float = 1.0
    hidden_units: tuple[int, ...] = (64, 64)
    activation: str = "elu"


# rl_games yaml key (under params.config) -> Args field name
RL_GAMES_KEYS = {
    "gamma": "gamma", "tau": "gae_lambda", "learning_rate": "learning_rate", "lr_schedule": "lr_schedule",
    "kl_threshold": "kl_threshold", "horizon_length": "num_steps", "minibatch_size": "minibatch_size",
    "mini_epochs": "update_epochs", "e_clip": "clip_coef", "entropy_coef": "ent_coef", "critic_coef": "vf_coef",
    "grad_norm": "max_grad_norm", "bounds_loss_coef": "bounds_loss_coef", "normalize_advantage": "norm_adv",
    "normalize_value": "normalize_value", "clip_value": "clip_vloss", "normalize_input": "normalize_input",
    "max_epochs": "max_epochs", "seq_length": "seq_length",
}


def _rl_games_flat(p):
    """params: section of an rl_games ppo yaml -> {Args field: value}."""
    flat = {RL_GAMES_KEYS[k]: v for k, v in p["config"].items() if k in RL_GAMES_KEYS}
    flat["reward_scale"] = p["config"].get("reward_shaper", {}).get("scale_value", 1.0)
    flat["seed"] = p.get("seed", 1)
    flat["clip_actions"] = p.get("env", {}).get("clip_actions", 1.0)
    flat["hidden_units"] = p.get("network", {}).get("mlp", {}).get("units", [64, 64])
    flat["activation"] = p.get("network", {}).get("mlp", {}).get("activation", "elu")
    return flat


def load_cfg_flat(path):
    """Read a config yaml -> {name: value}. Two formats:
    * an rl_games ppo yaml (has a `params:` section), translated with RL_GAMES_KEYS;
    * a flat yaml whose keys are Args field names, with an optional `include: other.yaml` (relative path) as base.
    """
    data = yaml.safe_load(open(path))
    if "params" in data:
        return _rl_games_flat(data["params"])
    flat = load_cfg_flat(os.path.join(os.path.dirname(path), data["include"])) if "include" in data else {}
    flat.update({k: v for k, v in data.items() if k != "include"})
    return flat


def load_cfg(path, Args):
    """Config yaml -> {Args field: value}, cast to the type of the field default. Unknown keys are ignored."""
    flat, defaults, out = load_cfg_flat(path), None, {}
    defaults = Args()
    for f in dataclasses.fields(Args):
        if f.name not in flat:
            continue
        d, v = getattr(defaults, f.name), flat[f.name]
        out[f.name] = tuple(type(d[0])(x) for x in v) if isinstance(d, tuple) else (v if isinstance(d, bool) else type(d)(v))
    return out


def parse_args(Args, algo):
    """Split Isaac Sim launcher flags (--headless, --device ...) from algorithm flags, launch the app.

    Defaults come from the dataclass, then configs/<algo>.yaml (or --cfg), then the command line.
    Returns (args, device, simulation_app). Must be called before importing anything from omni.*
    """
    import sys
    from omni.isaac.lab.app import AppLauncher

    sys.stdout.reconfigure(line_buffering=True)  # Kit may exit without flushing when stdout is redirected
    launcher = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    AppLauncher.add_app_launcher_args(launcher)
    app_args, rest = launcher.parse_known_args()
    default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{algo}.yaml")
    peek = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    peek.add_argument("--cfg", default=default_cfg if os.path.exists(default_cfg) else "")
    cfg_path = peek.parse_known_args(rest)[0].cfg
    base = dataclasses.replace(Args(), cfg=cfg_path, **(load_cfg(cfg_path, Args) if cfg_path else {}))
    args = tyro.cli(Args, args=rest, default=base)  # CLI flags beat the yaml
    app = AppLauncher(app_args).app
    return args, app_args.device, app


# ----------------------------------------------------------------------------- env
class IsaacEnv:
    """Thin torch-only wrapper: reset() -> obs, step(a) -> obs, reward, terminated, truncated.

    Isaac Lab auto-resets finished envs, so the obs returned on a done step is already the new episode's obs.
    """

    def __init__(self, task, num_envs, device, seed):
        import gymnasium as gym
        import omni.isaac.lab_tasks  # noqa: F401  (registers the tasks)
        from omni.isaac.lab_tasks.utils import parse_env_cfg

        cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
        cfg.seed = seed
        self.env = gym.make(task, cfg=cfg)
        self.u = self.env.unwrapped
        self.num_envs, self.device = num_envs, self.u.device
        self.obs_dim = gym.spaces.flatdim(self.u.single_observation_space["policy"])
        self.act_dim = gym.spaces.flatdim(self.u.single_action_space)
        self.info, self._last_log, self._log_sum, self._log_w = {}, None, {}, 0.0

    def reset(self):
        return self.env.reset()[0]["policy"]

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self.info = info  # raw env extras (eval mode: info["info"] holds per-step quantities)
        log = info.get("log")  # the env writes a fresh dict on every reset and keeps it until the next one
        if log is not None and log is not self._last_log:
            self._last_log = log
            w = float((term | trunc).sum())
            if w > 0:  # weight by the number of envs that finished in this step
                for k, v in log.items():
                    self._log_sum[k] = self._log_sum.get(k, 0.0) + float(v) * w
                self._log_w += w
        return obs["policy"], rew, term, trunc

    def drain_log(self):
        """Mean of the env's episode logs (Episode_Reward/<component>, Metrics/*, ...) since the last call."""
        out = {k: v / self._log_w for k, v in self._log_sum.items()} if self._log_w > 0 else {}
        self._log_sum, self._log_w = {}, 0.0
        return out

    def her_state(self):
        """Robot pose and goal (world frame, xy) -- only used by HER relabeling."""
        d = self.u._robot.data
        return dict(pos=d.root_link_pos_w[:, :2].clone(), yaw=d.heading_w.clone(), goal=self.u._desired_pos_w[:, :2].clone())

    def her_params(self):
        return dict(thr=self.u.cfg.goal_reached_threshold, max_dist=self.u.cfg.max_target_distance)

    def close(self):
        self.env.close()


# Action layout of the Kingfisher env: [thruster_left, thruster_right, rudder, sail]. Change it for another task.
ACTION_GROUPS = {"thrusters": [0, 1], "rudder": [2], "sail": [3]}


def action_bins(args):
    """Levels per action dimension from the bin_thrusters / bin_rudder / bin_sail settings (args or a checkpoint's args)."""
    get = (lambda k: args[k]) if isinstance(args, dict) else (lambda k: getattr(args, k))
    n = 1 + max(i for idx in ACTION_GROUPS.values() for i in idx)
    bins = [1] * n
    for group, idx in ACTION_GROUPS.items():
        for i in idx:
            bins[i] = int(get(f"bin_{group}"))
    return bins


class DiscreteEnv:
    """Discretizes the continuous [-1, 1]^d action space, with its own number of levels per dimension.

    bins[i] levels for dimension i, equally spaced in [-1, 1] (1 level = 0.0, 2 levels = -1 / +1, 3 = -1 / 0 / +1, ...).
    mode="multi": actions are (N, d) integers, action[:, i] in [0, bins[i])   -> PPO, one categorical per dimension
    mode="joint": actions are (N,) integers in [0, prod(bins))                 -> DQN / DRQN, one Q value per combination
    """

    def __init__(self, env, bins, mode="joint"):
        assert len(bins) == env.act_dim, f"bins {bins} must have one entry per action dimension ({env.act_dim})"
        self.env, self.bins, self.mode, self.act_dim = env, list(bins), mode, env.act_dim
        maxb = max(bins)
        self.levels = torch.zeros(len(bins), maxb, device=env.device)  # (d, maxb), padded
        for i, b in enumerate(bins):
            self.levels[i, :b] = torch.linspace(-1.0, 1.0, b) if b > 1 else 0.0
        self.strides = torch.tensor([int(np.prod(bins[:i])) for i in range(len(bins))], device=env.device)
        self.n_actions = int(np.prod(bins)) if mode == "joint" else int(sum(bins))
        self._dims = torch.arange(len(bins), device=env.device)

    def to_continuous(self, a):
        if self.mode == "joint":  # mixed-radix decode of the joint index
            a = (a.long().unsqueeze(-1) // self.strides) % torch.tensor(self.bins, device=a.device)
        return self.levels[self._dims, a.long()]

    def step(self, a):
        return self.env.step(self.to_continuous(a))

    def __getattr__(self, name):  # reset, obs_dim, act_dim, num_envs ... come from the wrapped env
        return getattr(self.env, name)


def make_env(args, device, bins=None, mode="joint"):
    """bins: list with the number of levels per action dimension (see action_bins) for a discretized env."""
    env = IsaacEnv(args.task, args.num_envs, device, args.seed)
    return DiscreteEnv(env, bins, mode) if bins else env


class MultiCategorical:
    """One categorical distribution per action dimension, with a different number of choices in each.

    logits: (B, d, maxb); entries beyond bins[i] are masked out. Actions are (B, d) integers.
    """

    def __init__(self, logits, bins):
        valid = torch.arange(logits.shape[-1], device=logits.device) < torch.tensor(bins, device=logits.device).unsqueeze(-1)
        self.dist = torch.distributions.Categorical(logits=logits.masked_fill(~valid, float("-inf")))

    def sample(self):
        return self.dist.sample()

    def mode(self):
        return self.dist.logits.argmax(-1)

    def log_prob(self, a):  # (B,)
        return self.dist.log_prob(a).sum(-1)

    def entropy(self):  # (B,)
        return self.dist.entropy().sum(-1)


# ----------------------------------------------------------------------------- logging
class Logger:
    """TensorBoard logger. Also writes the env's per-episode logs (every reward component, terminations, ...)
    by draining env.drain_log() at each log() call, so nothing has to be wired in the algorithm scripts."""

    def __init__(self, algo, args, env=None):
        from torch.utils.tensorboard import SummaryWriter

        self.dir = os.path.join(args.log_root, algo, args.task, args.exp_name or time.strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.dir, exist_ok=True)
        self.w = SummaryWriter(os.path.join(self.dir, "summaries"))
        yaml.safe_dump(dataclasses.asdict(args), open(os.path.join(self.dir, "args.yaml"), "w"))
        self.t0, self.args, self.env, self.algo = time.time(), args, env, algo
        print(f"[cleanrl_claude] logging to {self.dir}")

    def log(self, step, **kv):
        if self.env is not None:
            kv.update(self.env.drain_log())
        for k, v in kv.items():
            self.w.add_scalar(k, float(v), step)
        self.w.flush()

    def save(self, name, state):
        torch.save(state, os.path.join(self.dir, f"{name}.pt"))

    def save_model(self, name, state, **extra):
        """Standard checkpoint read by eval.py: algorithm name, all args, network weights (+ extras)."""
        self.save(name, dict(algo=self.algo, args=dataclasses.asdict(self.args), state=state, **extra))

    def sps(self, step):
        return step / (time.time() - self.t0)


class EpisodeStats:
    """Tracks episodic returns without a GPU sync on every step."""

    def __init__(self, num_envs, device, window=100):
        self.ret = torch.zeros(num_envs, device=device)
        self.returns = deque(maxlen=window)
        self.step_rewards = deque(maxlen=500)

    def update(self, rew, done):
        self.ret += rew
        self.step_rewards.append(rew.mean())
        if done.any():
            self.returns.extend(self.ret[done].tolist())
            self.ret[done] = 0.0

    def mean_return(self):
        return float(np.mean(self.returns)) if self.returns else float("nan")

    def mean_step_reward(self):
        return torch.stack(list(self.step_rewards)).mean().item() if self.step_rewards else float("nan")


def write_result(args, stats, timesteps):
    """Final numbers for the optuna tuner. The metric is the mean step reward when no episode finished."""
    out = dict(mean_return=stats.mean_return(), mean_step_reward=stats.mean_step_reward(), timesteps=timesteps)
    print("[cleanrl_claude] result:", out)
    if args.result_file:
        json.dump(out, open(args.result_file, "w"))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------- math
def mlp(sizes, act="elu", last_act=False):
    fn = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}[act]
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(fn())
    return nn.Sequential(*layers)


class RunningMeanStd:
    def __init__(self, shape=(), device="cpu"):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4

    def update(self, x):
        x = x.reshape(-1, *self.mean.shape)
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d**2 * self.count * bc / tot) / tot
        self.count = tot

    def normalize(self, x, clip=None):
        x = (x - self.mean) / torch.sqrt(self.var + 1e-8)
        return x.clamp(-clip, clip) if clip else x

    def denormalize(self, x):
        return x * torch.sqrt(self.var + 1e-8) + self.mean


def compute_gae(rew, val, done, next_val, gamma, lam):
    """rew, val, done: (T, N). done[t]=1 means the episode ended right after step t."""
    adv, last = torch.zeros_like(rew), 0.0
    for t in reversed(range(rew.shape[0])):
        nv = next_val if t == rew.shape[0] - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv


def adapt_lr(lr, kl, threshold):
    """rl_games 'adaptive' schedule."""
    if kl > 2.0 * threshold:
        return max(lr / 1.5, 1e-6)
    if kl < 0.5 * threshold:
        return min(lr * 1.5, 1e-2)
    return lr
