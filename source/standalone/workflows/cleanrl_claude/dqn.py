"""DQN for Isaac Lab. --bins gives the number of levels for each action dimension, in the env's own action
order, and every combination is one action (3 x 3 x 5 x 5 = 225 with the default --bins 3 3 5 5).

pysaac source/standalone/workflows/cleanrl_claude/dqn.py --headless --num_envs 64 --bins 3 3 5 5
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from buffers import ReplayBuffer
from common import BaseArgs, EpisodeStats, Logger, make_env, mlp, parse_args, seed_everything, write_result


@dataclass
class Args(BaseArgs):
    total_timesteps: int = 1_000_000
    bins: tuple[int, ...] = (3, 3, 5, 5)  # levels per action dimension, in the env's action order (Kingfisher: [thruster_left, thruster_right, rudder, sail])
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 5000
    updates_per_step: int = 1
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 1.0  # 1.0 = hard target copy every target_frequency updates
    target_frequency: int = 500
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.4  # of total_timesteps
    double_dqn: bool = True
    hidden: int = 256
    reward_scale: float = 1.0


def main():
    args, device, app = parse_args(Args, "dqn")
    seed_everything(args.seed)
    env = make_env(args, device, bins=args.bins, mode="joint")
    N, D, dev, n_act = env.num_envs, env.obs_dim, env.device, env.n_actions

    q = mlp([D, args.hidden, args.hidden, n_act], "relu").to(dev)
    q_t = mlp([D, args.hidden, args.hidden, n_act], "relu").to(dev)
    q_t.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=args.learning_rate)
    rb = ReplayBuffer(args.buffer_size, N, D, (), dev, act_dtype=torch.long)
    log, stats = Logger("dqn", args, env), EpisodeStats(N, dev)
    obs, step, updates = env.reset(), 0, 0

    while step < args.total_timesteps:
        eps = max(args.end_e, args.start_e + (args.end_e - args.start_e) * step / (args.exploration_fraction * args.total_timesteps))
        with torch.no_grad():
            greedy = q(obs).argmax(-1)
        rand = torch.randint(0, n_act, (N,), device=dev)
        action = torch.where(torch.rand(N, device=dev) < eps, rand, greedy)
        next_obs, rew, term, trunc = env.step(action)
        rb.add(obs, action, rew * args.reward_scale, term, trunc, next_obs)
        stats.update(rew, term | trunc)
        obs, step = next_obs, step + N

        if len(rb) < args.learning_starts:
            continue
        for _ in range(args.updates_per_step):
            b = rb.sample(args.batch_size)
            with torch.no_grad():
                if args.double_dqn:
                    best = q(b["next_obs"]).argmax(-1, keepdim=True)
                    next_q = q_t(b["next_obs"]).gather(-1, best).squeeze(-1)
                else:
                    next_q = q_t(b["next_obs"]).max(-1).values
                target = b["rew"] + args.gamma * (1 - b["term"]) * next_q
            pred = q(b["obs"]).gather(-1, b["act"].unsqueeze(-1)).squeeze(-1)
            loss = F.smooth_l1_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            updates += 1
            if updates % args.target_frequency == 0:
                for p, tp in zip(q.parameters(), q_t.parameters()):
                    tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

        if (step // N) % 50 == 0:
            log.log(step, **{"charts/sps": log.sps(step), "loss/td": loss.item(), "charts/epsilon": eps,
                             "charts/mean_step_reward": stats.mean_step_reward(), "charts/ep_return": stats.mean_return()})
            print(f"step {step} sps {log.sps(step):.0f} eps {eps:.2f} step_rew {stats.mean_step_reward():.4f} "
                  f"ep_ret {stats.mean_return():.2f} loss {loss.item():.4f}")
        if args.save_interval and step % args.save_interval < N:
            log.save_model(f"model_{step}", q.state_dict())

    log.save_model("model_last", q.state_dict())
    write_result(args, stats, step)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
