"""DDPG (continuous actions) for Isaac Lab, with an optional HER replay buffer (--her).

pysaac source/standalone/workflows/cleanrl_claude/ddpg.py --headless --num_envs 64
pysaac source/standalone/workflows/cleanrl_claude/ddpg.py --headless --her
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
    updates_per_step: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_frequency: int = 2
    exploration_noise: float = 0.1
    hidden: int = 256
    reward_scale: float = 1.0
    her: bool = False
    her_k: int = 4
    her_ratio: float = 0.8
    her_goal_reward: float = 1.0
    goal_obs_idx: tuple[int, int, int] = (6, 7, 8)


def main():
    args, device, app = parse_args(Args, "ddpg")
    seed_everything(args.seed)
    env = make_env(args, device)
    N, D, A, dev = env.num_envs, env.obs_dim, env.act_dim, env.device

    make_actor = lambda: nn.Sequential(mlp([D, args.hidden, args.hidden, A], "relu"), nn.Tanh()).to(dev)
    make_q = lambda: mlp([D + A, args.hidden, args.hidden, 1], "relu").to(dev)
    actor, actor_t, q, q_t = make_actor(), make_actor(), make_q(), make_q()
    actor_t.load_state_dict(actor.state_dict())
    q_t.load_state_dict(q.state_dict())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    q_opt = torch.optim.Adam(q.parameters(), lr=args.q_lr)

    her = HerCfg(**env.her_params(), k=args.her_k, ratio=args.her_ratio, goal_reward=args.her_goal_reward,
                 goal_obs_idx=args.goal_obs_idx) if args.her else None
    rb = ReplayBuffer(args.buffer_size, N, D, (A,), dev, her=her)
    log, stats = Logger("ddpg", args, env), EpisodeStats(N, dev)
    obs, step, updates = env.reset(), 0, 0

    while step < args.total_timesteps:
        state = env.her_state() if args.her else None
        if len(rb) < args.learning_starts:
            action = torch.rand(N, A, device=dev) * 2 - 1
        else:
            with torch.no_grad():
                action = (actor(obs) + torch.randn(N, A, device=dev) * args.exploration_noise).clamp(-1, 1)
        next_obs, rew, term, trunc = env.step(action)
        rb.add(obs, action, rew * args.reward_scale, term, trunc, next_obs, state)
        stats.update(rew, term | trunc)
        obs, step = next_obs, step + N

        if len(rb) < args.learning_starts:
            continue
        for _ in range(args.updates_per_step):
            b = rb.sample(args.batch_size)
            with torch.no_grad():
                nq = q_t(torch.cat([b["next_obs"], actor_t(b["next_obs"])], -1))
                target = b["rew"].unsqueeze(-1) + (1 - b["term"].unsqueeze(-1)) * args.gamma * nq
            q_loss = F.mse_loss(q(torch.cat([b["obs"], b["act"]], -1)), target)
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()
            updates += 1
            if updates % args.policy_frequency == 0:
                actor_loss = -q(torch.cat([b["obs"], actor(b["obs"])], -1)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                with torch.no_grad():
                    for net, tgt in ((actor, actor_t), (q, q_t)):
                        for p, tp in zip(net.parameters(), tgt.parameters()):
                            tp.mul_(1 - args.tau).add_(args.tau * p)

        if (step // N) % 50 == 0:
            log.log(step, **{"charts/sps": log.sps(step), "loss/q": q_loss.item(),
                             "charts/mean_step_reward": stats.mean_step_reward(), "charts/ep_return": stats.mean_return()})
            print(f"step {step} sps {log.sps(step):.0f} step_rew {stats.mean_step_reward():.4f} "
                  f"ep_ret {stats.mean_return():.2f} q_loss {q_loss.item():.4f}")
        if args.save_interval and step % args.save_interval < N:
            log.save_model(f"model_{step}", actor.state_dict(), critic=q.state_dict())

    log.save_model("model_last", actor.state_dict(), critic=q.state_dict())
    write_result(args, stats, step)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
