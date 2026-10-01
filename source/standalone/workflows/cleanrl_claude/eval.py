"""Evaluate any cleanrl_claude checkpoint on the paper sweep (wind angle x wind speed x energy context).

Does what rl_games/icra2027/ours/play_kingfisher_paper_eval.py does, for every algorithm of this directory:
each env runs EPISODES_PER_ENV = n_wind_angles * n_wind_speeds episodes, one row per time step is written to
all_steps_swept_<run>.csv (same columns, so plot_paper_eval_3D.py reads it unchanged), and a summary
(success rate, time, energy, per-context / per-wind-speed success, mean reward components) is printed, saved as
eval_summary.json next to the csv and written to TensorBoard (<checkpoint dir>/eval).

pysaac source/standalone/workflows/cleanrl_claude/eval.py --headless --num_envs 10 \
    --checkpoint logs/cleanrl_claude/ppo/Isaac-KingfisherSail-Direct-v0/<run>/model_last.pt

CSV location: --csv_path if given; else next to KINGFISHER_REWARD_CFG (as play_kingfisher_paper_eval.py does) when
that variable is set; else next to the checkpoint.
"""
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

import common
from common import BaseArgs, make_env, parse_args, seed_everything


@dataclass
class Args(BaseArgs):
    checkpoint: str = ""  # path to a model_*.pt written by the training scripts
    task: str = ""  # default: the task stored in the checkpoint
    num_envs: int = 10
    csv_path: str = ""
    episodes_per_env: int = 0  # 0 = the full sweep (n_angles * n_speeds); smaller = quick test
    stochastic: bool = False  # sample actions (PPO / SAC only) instead of the deterministic policy


# ----------------------------------------------------------------------------- policies
def build_policy(ck, env, dev, stochastic=False):
    """Returns (act_fn, reset_fn). act_fn(obs) -> action for the env (continuous, or integers for discrete algos);
    reset_fn(done) clears the hidden state of finished envs (recurrent policies)."""
    import drqn, ppo, ppo_discrete, ppo_rnd, ppo_rnn, sac  # (Args classes only; nothing runs at import)

    a, algo, st = SimpleNamespace(**ck["args"]), ck["algo"], ck["state"]
    D, A = env.obs_dim, env.act_dim
    no_reset = lambda done: None
    if "obs_rms" in ck and getattr(a, "normalize_input", False):
        mean, var = (t.to(dev) for t in ck["obs_rms"])
        norm = lambda x: ((x - mean) / torch.sqrt(var + 1e-8)).clamp(-10.0, 10.0)
    else:
        norm = lambda x: x

    if algo in ("ppo", "ppo_rnd", "ppo_rnd_rnn"):  # the two rnd variants share the ppo_rnd actor-critic
        m = (ppo if algo == "ppo" else ppo_rnd).Agent(D, A, tuple(a.hidden_units), a.activation).to(dev)
        m.load_state_dict(st)
        def act(obs):
            mu = m(norm(obs))[0]
            a_ = torch.distributions.Normal(mu, m.logstd.exp()).sample() if stochastic else mu
            return a_.clamp(-a.clip_actions, a.clip_actions)
        return act, no_reset
    if algo == "ppo_discrete":
        m = ppo_discrete.Agent(D, A, ck['args']['bins'], tuple(a.hidden_units), a.activation).to(dev)
        m.load_state_dict(st)
        return (lambda obs: (m(norm(obs))[0].sample() if stochastic else m(norm(obs))[0].mode())), no_reset
    if algo == "ppo_rnn":
        m = ppo_rnn.Agent(D, A, tuple(a.hidden_units), a.activation, a.lstm_hidden).to(dev)
        m.load_state_dict(st)
        mem = dict(state=m.zero_state(env.num_envs, dev), start=torch.zeros(env.num_envs, device=dev))
        def act(obs):
            mu, _, mem["state"] = m(norm(obs).unsqueeze(0), mem["state"], mem["start"].unsqueeze(0))
            mu = mu[0]
            a_ = torch.distributions.Normal(mu, m.logstd.exp()).sample() if stochastic else mu
            return a_.clamp(-a.clip_actions, a.clip_actions)
        def reset(done):
            mem["start"] = done.float()
        return act, reset
    if algo == "sac":
        m = sac.Actor(D, A, a.hidden).to(dev)
        m.load_state_dict(st)
        return (lambda obs: m.sample(obs)[0] if stochastic else torch.tanh(m(obs)[0])), no_reset
    if algo == "ddpg":
        m = nn.Sequential(common.mlp([D, a.hidden, a.hidden, A], "relu"), nn.Tanh()).to(dev)
        m.load_state_dict(st)
        return (lambda obs: m(obs)), no_reset
    if algo == "dqn":
        m = common.mlp([D, a.hidden, a.hidden, int(np.prod(ck["args"]["bins"]))], "relu").to(dev)
        m.load_state_dict(st)
        return (lambda obs: m(obs).argmax(-1)), no_reset
    if algo == "drqn":
        m = drqn.RecurrentQNet(D, int(np.prod(ck["args"]["bins"])), a.hidden).to(dev)
        m.load_state_dict(st)
        mem = dict(state=m.zero_state(env.num_envs, dev))
        def act(obs):
            q, mem["state"] = m.step(obs, mem["state"])
            return q.argmax(-1)
        def reset(done):
            keep = (1 - done.float()).unsqueeze(-1)
            mem["state"] = (mem["state"][0] * keep, mem["state"][1] * keep)
        return act, reset
    raise ValueError(f"unknown algorithm '{algo}' in checkpoint")


# ----------------------------------------------------------------------------- helpers
def resolve_csv_path(args, ckpt_dir):
    if args.csv_path:
        return args.csv_path
    cfg = os.environ.get("KINGFISHER_REWARD_CFG")
    run_dir = os.path.dirname(os.path.abspath(cfg)) if cfg else ckpt_dir
    return os.path.join(run_dir, f"all_steps_swept_{os.path.basename(run_dir)}.csv")


def summarize(df, step_dt):
    """Per-episode table -> success / time / energy overall and per context / wind speed."""
    ep = df.groupby(["env_id", "local_episode_idx"]).agg(
        success=("goal_reached", "first"), timed_out=("timed_out", "first"), steps=("time_step", "max"),
        energy=("energy", "sum"), context=("energy_context", "first"), wind_speed=("true_wind_speed", "first"),
        wind_angle=("true_wind_angle_w", "first")).reset_index()
    ok = ep[ep.success]
    out = dict(
        episodes=len(ep), success_rate=float(ep.success.mean()), timeout_rate=float(ep.timed_out.mean()),
        mean_time_to_goal_s=float(ok.steps.mean() * step_dt) if len(ok) else None,
        mean_energy_success=float(ok.energy.mean()) if len(ok) else None,
        success_by_context={f"{k:.2f}": float(v) for k, v in ep.groupby(ep.context.round(2)).success.mean().items()},
        success_by_wind_speed={f"{k:g}": float(v) for k, v in ep.groupby(ep.wind_speed.round(2)).success.mean().items()},
        success_by_wind_angle={f"{k:g}": float(v) for k, v in ep.groupby(ep.wind_angle.round(1)).success.mean().items()},
    )
    return out


def main():
    args, device, app = parse_args(Args, "eval")
    import pandas as pd
    from torch.utils.tensorboard import SummaryWriter

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    algo = ck["algo"]
    args.task = args.task or ck["args"]["task"]
    seed_everything(args.seed)
    discrete = algo in ("ppo_discrete", "dqn", "drqn")
    env = make_env(args, device, bins=ck["args"]["bins"] if discrete else None, mode="multi" if algo == "ppo_discrete" else "joint")
    act, reset_policy = build_policy(ck, env, env.device, args.stochastic)
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    csv_path = resolve_csv_path(args, ckpt_dir)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    u, N = env.u, env.num_envs
    episodes_per_env = args.episodes_per_env or u.paper_wind_angle.shape[1] * u.paper_wind_speed.shape[1]
    u.is_Training = False
    u.is_paper_eval = True
    obs = env.reset()
    print(f"[eval] {algo}: {episodes_per_env} episodes/env x {N} envs = {episodes_per_env * N} episodes -> {csv_path}")

    done_count = np.zeros(N, dtype=int)
    buffers = [[] for _ in range(N)]
    rows = []

    def save():
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"[eval] saved '{csv_path}': {len(df)} rows, {df.groupby(['env_id', 'local_episode_idx']).ngroups} episodes")
        return df

    def np1(info, key, default=0.0):  # (N,) numpy column from the env info (zeros when the env does not provide it)
        v = info.get(key)
        return np.full(N, default, dtype=np.float32) if v is None else v.detach().reshape(N, -1)[:, 0].float().cpu().numpy()

    try:
        while not (done_count >= episodes_per_env).all():
            with torch.inference_mode():
                action = act(obs)
                obs, rew, term, trunc = env.step(action)
            done = term | trunc
            reset_policy(done)
            info = env.info.get("info", {})
            pos = info["robot_pos_w"][:, :2].detach().cpu().numpy() if "robot_pos_w" in info else np.zeros((N, 2))
            goal = info["goal_pos"][:, :2].detach().cpu().numpy() if "goal_pos" in info else np.zeros((N, 2))
            aero = info["aero_force"].detach().reshape(N, -1)[:, 0].cpu().numpy() if "aero_force" in info else np.zeros(N)
            cols = {k: np1(info, k) for k in (
                "energy", "distance", "energy_context", "true_wind_speed", "true_wind_angle_w", "sail_angle", "aoa",
                "app_wind_angle", "true_wind_angle_b", "reward_progress", "reward_goal", "reward_energy",
                "reward_backward", "reward_bearing", "reward_aero", "reward_acord")}
            done_np = done.cpu().numpy()
            terminated, timed_out = u.reset_terminated.cpu().numpy(), u.reset_time_outs.cpu().numpy()
            for i in range(N):
                if done_count[i] >= episodes_per_env:
                    continue
                row = dict(env_id=i, time_step=len(buffers[i]), energy=cols["energy"][i], distance=cols["distance"][i],
                           energy_context=cols["energy_context"][i], true_wind_speed=cols["true_wind_speed"][i],
                           true_wind_angle_w=cols["true_wind_angle_w"][i], rb_pos_x=pos[i, 0], rb_pos_y=pos[i, 1],
                           sail_angle=cols["sail_angle"][i], aero_force=aero[i], aoa=cols["aoa"][i],
                           app_wind_angle=cols["app_wind_angle"][i], true_wind_angle_b=cols["true_wind_angle_b"][i],
                           g_pos_x=goal[i, 0], g_pos_y=goal[i, 1])
                row.update({k: cols[k][i] for k in cols if k.startswith("reward_")})
                if done_np[i]:
                    # this row is already the NEXT episode's first state (auto reset): keep it as t=0 of the next
                    # episode, and read the outcome from the sim's own flags (as play_kingfisher_paper_eval.py does)
                    for r in buffers[i]:
                        r.update(local_episode_idx=int(done_count[i]), goal_reached=bool(terminated[i]), timed_out=bool(timed_out[i]))
                    rows.extend(buffers[i])
                    done_count[i] += 1
                    row["time_step"] = 0
                    buffers[i] = [row]
                    if done_count.sum() % 10 == 0:
                        print(f"[eval] episodes finished: {done_count.sum()} {done_count}")
                else:
                    buffers[i].append(row)
    except KeyboardInterrupt:
        print("[eval] interrupted, saving what was collected")

    df = save()
    if df.empty:
        print("[eval] no finished episode -- nothing to summarize")
    else:
        summary = summarize(df, u.step_dt)
        summary["reward_components_per_episode"] = {k: v for k, v in env.drain_log().items() if k.startswith("Episode_Reward/")}
        summary.update(algo=algo, checkpoint=os.path.abspath(args.checkpoint), csv=os.path.abspath(csv_path))
        json.dump(summary, open(os.path.join(os.path.dirname(csv_path), "eval_summary.json"), "w"), indent=2)
        print(json.dumps(summary, indent=2))
        w = SummaryWriter(os.path.join(ckpt_dir, "eval"))
        for k in ("success_rate", "timeout_rate", "mean_time_to_goal_s", "mean_energy_success"):
            if summary[k] is not None:
                w.add_scalar(f"eval/{k}", summary[k], 0)
        for name in ("success_by_context", "success_by_wind_speed", "success_by_wind_angle"):
            for k, v in summary[name].items():
                w.add_scalar(f"eval/{name}/{k}", v, 0)
        for k, v in summary["reward_components_per_episode"].items():
            w.add_scalar(f"eval/{k}", v, 0)
        w.close()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
