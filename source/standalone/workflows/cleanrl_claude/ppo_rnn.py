"""PPO with an LSTM policy (recurrent PPO) for Isaac Lab. Same yaml / hyper-parameters as ppo.py.

The rollout is cut into sequences of --seq_length steps. Each sequence starts from the LSTM state stored during the
rollout, is unrolled over time, and the state is reset at episode boundaries (as in CleanRL's ppo_atari_lstm.py).
Mini-batches are made of whole sequences.

pysaac source/standalone/workflows/cleanrl_claude/ppo_rnn.py --headless --num_envs 4096
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Normal

from common import (PPOArgs, Logger, EpisodeStats, RunningMeanStd, adapt_lr, compute_gae, make_env, mlp, parse_args,
                    seed_everything, write_result)


@dataclass
class Args(PPOArgs):
    lstm_hidden: int = 64
    seq_length: int = 8  # steps the LSTM is unrolled (BPTT) per training sequence; must divide num_steps


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, units, act, hidden):
        super().__init__()
        self.hidden = hidden
        self.torso = mlp([obs_dim, *units], act, last_act=True)
        self.lstm = nn.LSTM(units[-1], hidden)
        self.mu, self.value = nn.Linear(hidden, act_dim), nn.Linear(hidden, 1)
        self.logstd = nn.Parameter(torch.zeros(act_dim))

    def zero_state(self, batch, device):
        return torch.zeros(1, batch, self.hidden, device=device), torch.zeros(1, batch, self.hidden, device=device)

    def forward(self, obs, state, start):
        """obs (T, B, D); start (T, B) = 1 where obs is the first of an episode (state is reset there)."""
        x, outs = self.torso(obs), []
        for xt, st in zip(x, start):
            m = (1.0 - st).view(1, -1, 1)
            o, state = self.lstm(xt.unsqueeze(0), (state[0] * m, state[1] * m))
            outs.append(o)
        h = torch.cat(outs)
        return self.mu(h), self.value(h).squeeze(-1), state


def main():
    args, device, app = parse_args(Args, "ppo_rnn")
    seed_everything(args.seed)
    env = make_env(args, device)
    N, T, D, A, dev = env.num_envs, args.num_steps, env.obs_dim, env.act_dim, env.device
    batch = N * T
    L = max(d for d in range(1, min(args.seq_length, T) + 1) if T % d == 0)  # largest divisor of num_steps <= seq_length
    if L != args.seq_length:
        print(f"[ppo_rnn] seq_length {args.seq_length} does not divide num_steps {T}: using {L}")
    n_chunks, seqs_per_mb = T // L, max(min(args.minibatch_size, batch) // L, 1)
    iters = args.total_timesteps // batch if args.total_timesteps else args.max_epochs

    agent = Agent(D, A, args.hidden_units, args.activation, args.lstm_hidden).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    obs_rms, val_rms = RunningMeanStd((D,), dev), RunningMeanStd((), dev)
    norm = lambda x: obs_rms.normalize(x, 10.0) if args.normalize_input else x
    denorm_v = lambda v: val_rms.denormalize(v) if args.normalize_value else v
    norm_v = lambda v: val_rms.normalize(v) if args.normalize_value else v

    log, stats, lr = Logger("ppo_rnn", args, env), EpisodeStats(N, dev), args.learning_rate
    z = lambda *s: torch.zeros(T, N, *s, device=dev)
    b_obs, b_act, b_logp, b_rew, b_done, b_val, b_start = z(D), z(A), z(), z(), z(), z(), z()
    b_h, b_c = z(args.lstm_hidden), z(args.lstm_hidden)  # LSTM state before each step (start state of every sequence)
    obs, step = env.reset(), 0
    state, start = agent.zero_state(N, dev), torch.zeros(N, device=dev)

    for it in range(iters):
        if args.lr_schedule == "linear":
            lr = args.learning_rate * (1 - it / iters)
            opt.param_groups[0]["lr"] = lr
        for t in range(T):
            if args.normalize_input:
                obs_rms.update(obs)
            b_h[t], b_c[t] = state[0][0], state[1][0]
            with torch.no_grad():
                mu, v, state = agent(norm(obs).unsqueeze(0), state, start.unsqueeze(0))
                dist = Normal(mu[0], agent.logstd.exp())
                action, v = dist.sample(), denorm_v(v[0])
            next_obs, rew, term, trunc = env.step(action.clamp(-args.clip_actions, args.clip_actions))
            done = (term | trunc).float()
            r = rew * args.reward_scale + args.gamma * v * trunc.float()
            b_obs[t], b_act[t], b_logp[t], b_rew[t], b_done[t], b_val[t], b_start[t] = \
                obs, action, dist.log_prob(action).sum(-1), r, done, v, start
            stats.update(rew, term | trunc)
            obs, start, step = next_obs, done, step + N

        with torch.no_grad():
            next_v = denorm_v(agent(norm(obs).unsqueeze(0), state, start.unsqueeze(0))[1][0])
        adv = compute_gae(b_rew, b_val, b_done, next_v, args.gamma, args.gae_lambda)
        ret = adv + b_val
        if args.normalize_value:
            val_rms.update(ret)
        # Cut the rollout into sequences of L steps: (T, N, ...) -> (L, n_chunks * N, ...). Sequence k * N + e is
        # steps k*L .. (k+1)*L - 1 of env e, and starts from the LSTM state that was stored at step k*L (b_h[::L]).
        seq = lambda x: x.reshape(n_chunks, L, N, *x.shape[2:]).transpose(0, 1).reshape(L, n_chunks * N, *x.shape[2:])
        s_obs, s_act, s_logp, s_start, s_adv = seq(norm(b_obs)), seq(b_act), seq(b_logp), seq(b_start), seq(adv)
        s_ret, s_val = seq(norm_v(ret)), seq(norm_v(b_val))
        s_h, s_c = b_h[::L].reshape(1, n_chunks * N, -1), b_c[::L].reshape(1, n_chunks * N, -1)

        for _ in range(args.update_epochs):
            perm, kls = torch.randperm(n_chunks * N, device=dev), []
            for s in range(0, n_chunks * N, seqs_per_mb):
                e = perm[s:s + seqs_per_mb]
                mu, v, _ = agent(s_obs[:, e], (s_h[:, e], s_c[:, e]), s_start[:, e])
                dist = Normal(mu, agent.logstd.exp())
                logratio = dist.log_prob(s_act[:, e]).sum(-1) - s_logp[:, e]
                ratio = logratio.exp().reshape(-1)
                kls.append(((ratio - 1) - logratio.reshape(-1)).mean().detach())
                a = s_adv[:, e].reshape(-1)
                a = (a - a.mean()) / (a.std() + 1e-8) if args.norm_adv else a
                pg_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
                v, tgt, old = v.reshape(-1), s_ret[:, e].reshape(-1), s_val[:, e].reshape(-1)
                if args.clip_vloss:
                    v_clip = old + (v - old).clamp(-args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max((v - tgt) ** 2, (v_clip - tgt) ** 2).mean()
                else:
                    v_loss = 0.5 * ((v - tgt) ** 2).mean()
                bound_loss = ((mu - 1).clamp(min=0) ** 2 + (-mu - 1).clamp(min=0) ** 2).sum(-1).mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * dist.entropy().sum(-1).mean() \
                    + args.bounds_loss_coef * bound_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                opt.step()
            kl = torch.stack(kls).mean().item()
            if args.lr_schedule == "adaptive":
                lr = adapt_lr(lr, kl, args.kl_threshold)
                opt.param_groups[0]["lr"] = lr

        log.log(step, **{"charts/sps": log.sps(step), "charts/lr": lr, "charts/approx_kl": kl,
                         "charts/mean_step_reward": stats.mean_step_reward(), "charts/ep_return": stats.mean_return(),
                         "loss/policy": pg_loss.item(), "loss/value": v_loss.item()})
        print(f"it {it + 1}/{iters} step {step} sps {log.sps(step):.0f} step_rew {stats.mean_step_reward():.4f} "
              f"ep_ret {stats.mean_return():.2f} lr {lr:.2e} kl {kl:.4f}")
        if args.save_interval and step % args.save_interval < batch:
            log.save_model(f"model_{step}", agent.state_dict())

    log.save_model("model_last", agent.state_dict(), obs_rms=(obs_rms.mean, obs_rms.var))
    write_result(args, stats, step)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
