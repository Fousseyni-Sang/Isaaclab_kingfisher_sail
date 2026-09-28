"""SAC (continuous actions) for Isaac Lab, with an optional HER replay buffer (--her).

pysaac source/standalone/workflows/cleanrl_claude/sac.py --headless --num_envs 64
pysaac source/standalone/workflows/cleanrl_claude/sac.py --headless --her --her_k 4   # goal-conditioned baseline
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from buffers import HerCfg, ReplayBuffer
from common import BaseArgs, EpisodeStats, Logger, make_env, mlp, parse_args, seed_everything, write_result


@dataclass
class Args(BaseArgs):
    total_timesteps: int = 1_000_000
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 5000
    updates_per_step: int = 1  # gradient updates per vectorized env step (each step gives num_envs transitions)
    gamma: float = 0.99
    tau: float = 0.005
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    alpha: float = 0.2
    autotune: bool = True
    hidden: int = 256
    reward_scale: float = 1.0
    her: bool = False  # Hindsight Experience Replay: sparse goal reward + goals relabeled from future positions
    her_k: int = 4
    her_ratio: float = 0.8
    her_goal_reward: float = 1.0
    goal_obs_idx: tuple[int, int, int] = (6, 7, 8)  # obs slots [cos bearing, sin bearing, dist / max_dist]


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, h):
        super().__init__()
        self.net = mlp([obs_dim, h, h], "relu", last_act=True)
        self.mu, self.logstd = nn.Linear(h, act_dim), nn.Linear(h, act_dim)

    def forward(self, x):
        h = self.net(x)
        logstd = -5 + 3.5 * (torch.tanh(self.logstd(h)) + 1)  # in [-5, 2]
        return self.mu(h), logstd

    def sample(self, x):
        mu, logstd = self(x)
        normal = torch.distributions.Normal(mu, logstd.exp())
        u = normal.rsample()
        a = torch.tanh(u)
        logp = (normal.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a, logp


def main():
    args, device, app = parse_args(Args, "sac")
    seed_everything(args.seed)
    env = make_env(args, device)
    N, D, A = env.num_envs, env.obs_dim, env.act_dim
    dev = env.device

    actor = Actor(D, A, args.hidden).to(dev)
    q = nn.ModuleList([mlp([D + A, args.hidden, args.hidden, 1], "relu") for _ in range(2)]).to(dev)
    q_t = nn.ModuleList([mlp([D + A, args.hidden, args.hidden, 1], "relu") for _ in range(2)]).to(dev)
    q_t.load_state_dict(q.state_dict())
    q_opt = torch.optim.Adam(q.parameters(), lr=args.q_lr)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.policy_lr)
    if args.autotune:
        target_entropy = -float(A)
        log_alpha = torch.zeros(1, requires_grad=True, device=dev)
        alpha_opt = torch.optim.Adam([log_alpha], lr=args.q_lr)
        alpha = log_alpha.exp().item()
    else:
        alpha = args.alpha

    her = HerCfg(**env.her_params(), k=args.her_k, ratio=args.her_ratio, goal_reward=args.her_goal_reward,
                 goal_obs_idx=args.goal_obs_idx) if args.her else None
    rb = ReplayBuffer(args.buffer_size, N, D, (A,), dev, her=her)
    log, stats = Logger("sac", args, env), EpisodeStats(N, dev)
    obs, step, updates = env.reset(), 0, 0

    while step < args.total_timesteps:
        state = env.her_state() if args.her else None
        if len(rb) < args.learning_starts:
            action = torch.rand(N, A, device=dev) * 2 - 1
        else:
            with torch.no_grad():
                action = actor.sample(obs)[0]
        next_obs, rew, term, trunc = env.step(action)
        rb.add(obs, action, rew * args.reward_scale, term, trunc, next_obs, state)
        stats.update(rew, term | trunc)
        obs, step = next_obs, step + N

        if len(rb) < args.learning_starts:
            continue
        for _ in range(args.updates_per_step):
            b = rb.sample(args.batch_size)
            with torch.no_grad():
                na, nlogp = actor.sample(b["next_obs"])
                x = torch.cat([b["next_obs"], na], -1)
                min_q = torch.min(q_t[0](x), q_t[1](x)) - alpha * nlogp
                target = b["rew"].unsqueeze(-1) + (1 - b["term"].unsqueeze(-1)) * args.gamma * min_q
            xa = torch.cat([b["obs"], b["act"]], -1)
            q_loss = F.mse_loss(q[0](xa), target) + F.mse_loss(q[1](xa), target)
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()
            updates += 1

            # delayed policy update (TD3 style): the actor and alpha are updated every policy_frequency critic updates,
            # policy_frequency times in a row to make up for the skipped steps (as in CleanRL)
            if updates % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    a, logp = actor.sample(b["obs"])
                    x = torch.cat([b["obs"], a], -1)
                    actor_loss = (alpha * logp - torch.min(q[0](x), q[1](x))).mean()
                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()
                    if args.autotune:
                        with torch.no_grad():
                            _, logp = actor.sample(b["obs"])
                        alpha_loss = (-log_alpha.exp() * (logp + target_entropy)).mean()
                        alpha_opt.zero_grad()
                        alpha_loss.backward()
                        alpha_opt.step()
                        alpha = log_alpha.exp().item()
            with torch.no_grad():
                for p, tp in zip(q.parameters(), q_t.parameters()):
                    tp.mul_(1 - args.tau).add_(args.tau * p)

        if (step // N) % 50 == 0:
            log.log(step, **{"charts/sps": log.sps(step), "loss/q": q_loss.item(), "loss/alpha": alpha,
                             "charts/mean_step_reward": stats.mean_step_reward(), "charts/ep_return": stats.mean_return()})
            print(f"step {step} sps {log.sps(step):.0f} step_rew {stats.mean_step_reward():.4f} "
                  f"ep_ret {stats.mean_return():.2f} q_loss {q_loss.item():.4f} alpha {alpha:.3f}")
        if args.save_interval and step % args.save_interval < N:
            log.save_model(f"model_{step}", actor.state_dict(), critic=q.state_dict())

    log.save_model("model_last", actor.state_dict(), critic=q.state_dict())
    write_result(args, stats, step)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
