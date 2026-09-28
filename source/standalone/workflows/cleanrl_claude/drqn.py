"""DRQN (recurrent DQN) for Isaac Lab, on the discretized action space (see dqn.py).

Training samples sequences of --seq_len steps from the replay buffer and starts the LSTM from a zero state;
the first --burn_in steps of every sequence only warm up the hidden state (no loss), as in R2D2-lite.

pysaac source/standalone/workflows/cleanrl_claude/drqn.py --headless --num_envs 64 --bin_rudder 5 --bin_sail 5 --bin_thrusters 3
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from buffers import ReplayBuffer
from common import BaseArgs, action_bins, EpisodeStats, Logger, make_env, mlp, parse_args, seed_everything, write_result


@dataclass
class Args(BaseArgs):
    total_timesteps: int = 1_000_000
    bin_thrusters: int = 3
    bin_rudder: int = 5
    bin_sail: int = 5
    buffer_size: int = 1_000_000
    batch_size: int = 64  # sequences per update
    seq_len: int = 16
    burn_in: int = 4
    learning_starts: int = 5000
    updates_per_step: int = 1
    learning_rate: float = 1e-4
    gamma: float = 0.99
    target_frequency: int = 500
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.4
    hidden: int = 128
    reward_scale: float = 1.0


class RecurrentQNet(nn.Module):
    def __init__(self, obs_dim, n_actions, h):
        super().__init__()
        self.h = h
        self.enc, self.cell, self.out = mlp([obs_dim, h], "relu", last_act=True), nn.LSTMCell(h, h), nn.Linear(h, n_actions)

    def step(self, obs, state):
        h, c = self.cell(self.enc(obs), state)
        return self.out(h), (h, c)

    def zero_state(self, batch, device):
        return torch.zeros(batch, self.h, device=device), torch.zeros(batch, self.h, device=device)

    def unroll(self, obs, done):
        """obs (B, L+1, D), done (B, L): done[:, k] = episode ended after step k -> reset the state before obs k+1."""
        state, qs = self.zero_state(obs.shape[0], obs.device), []
        for k in range(obs.shape[1]):
            if k > 0:
                m = (1 - done[:, k - 1]).unsqueeze(-1)
                state = (state[0] * m, state[1] * m)
            q, state = self.step(obs[:, k], state)
            qs.append(q)
        return torch.stack(qs, 1)


def main():
    args, device, app = parse_args(Args, "drqn")
    seed_everything(args.seed)
    env = make_env(args, device, bins=action_bins(args), mode="joint")
    N, D, dev, n_act = env.num_envs, env.obs_dim, env.device, env.n_actions

    q, q_t = RecurrentQNet(D, n_act, args.hidden).to(dev), RecurrentQNet(D, n_act, args.hidden).to(dev)
    q_t.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=args.learning_rate)
    rb = ReplayBuffer(args.buffer_size, N, D, (), dev, act_dtype=torch.long)
    log, stats = Logger("drqn", args, env), EpisodeStats(N, dev)
    obs, step, updates, state = env.reset(), 0, 0, q.zero_state(N, dev)

    while step < args.total_timesteps:
        eps = max(args.end_e, args.start_e + (args.end_e - args.start_e) * step / (args.exploration_fraction * args.total_timesteps))
        with torch.no_grad():
            qv, state = q.step(obs, state)
        action = torch.where(torch.rand(N, device=dev) < eps, torch.randint(0, n_act, (N,), device=dev), qv.argmax(-1))
        next_obs, rew, term, trunc = env.step(action)
        rb.add(obs, action, rew * args.reward_scale, term, trunc, next_obs)
        stats.update(rew, term | trunc)
        m = (1 - (term | trunc).float()).unsqueeze(-1)  # new episode -> fresh hidden state
        state = (state[0] * m, state[1] * m)
        obs, step = next_obs, step + N

        if len(rb) < args.learning_starts or rb.size <= args.seq_len + 1:
            continue
        for _ in range(args.updates_per_step):
            b = rb.sample_sequences(args.batch_size, args.seq_len)
            # Q-values of all L+1 observations of each window, computed with the LSTM unrolled from a zero state.
            # Step k is trained on obs[k] -> obs[k+1] (obs[k+1] is the next state unless the episode ended: then it is
            # the reset obs, but the target is cut by term/the state reset, so it does not matter).
            q_all = q.unroll(b["obs"], b["done"])  # (B, L+1, A)
            with torch.no_grad():
                q_next_t = q_t.unroll(b["obs"], b["done"])[:, 1:]
                best = q_all[:, 1:].argmax(-1, keepdim=True)  # double DQN: online net picks the action, target net scores it
                target = b["rew"] + args.gamma * (1 - b["term"]) * q_next_t.gather(-1, best).squeeze(-1)
            pred = q_all[:, :-1].gather(-1, b["act"].unsqueeze(-1)).squeeze(-1)
            # the first burn_in steps only warm up the hidden state (it starts from zero, not from the real history)
            loss = F.smooth_l1_loss(pred[:, args.burn_in:], target[:, args.burn_in:])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
            updates += 1
            if updates % args.target_frequency == 0:
                q_t.load_state_dict(q.state_dict())

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
