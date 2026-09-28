"""PPO + sequence-based Random Network Distillation.

ppo_rnd.py measures novelty of a single state. Here the target and the predictor are recurrent networks (LSTM, or GRU
with --rnd_cell gru) that read a whole window of the trajectory: the rollout of every env is cut into windows of
--rnd_seq_len steps (0 = the whole horizon num_steps), and at every step the predictor must reproduce the output
of the frozen random target given all the states of the window seen so far. The intrinsic reward of a step is that
prediction error, so it is high for state SEQUENCES the predictor has not seen, not only for new single states.
The recurrent state is zero at the start of a window and is reset at episode boundaries (target and predictor
identically, so the target stays a deterministic function of the window).

The policy is the same MLP actor-critic as ppo_rnd.py (two value heads: extrinsic + intrinsic). Same yaml as ppo.py.

pysaac source/standalone/workflows/cleanrl_claude/ppo_rnd_rnn.py --headless --num_envs 4096
pysaac source/standalone/workflows/cleanrl_claude/ppo_rnd_rnn.py --headless --rnd_cell gru --rnd_seq_len 24
"""
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch.distributions import Normal

from common import (Logger, EpisodeStats, RunningMeanStd, adapt_lr, compute_gae, make_env, parse_args,
                    seed_everything, write_result)
from ppo_rnd import Agent, Args as RNDArgs


@dataclass
class Args(RNDArgs):
    rnd_cell: Literal["lstm", "gru"] = "lstm"
    rnd_seq_len: int = 0  # window length in steps; 0 = num_steps (the whole horizon). Must divide num_steps.
    rnd_batch_seqs: int = 1024  # windows per predictor mini-batch


class SeqNet(nn.Module):
    """Recurrent net: (L, B, D) states -> (L, B, out), one output per step; hidden state reset where reset == 1."""

    def __init__(self, obs_dim, hidden, out, cell, deep_head=False, random_init=False):
        super().__init__()
        self.hidden, self.lstm = hidden, cell == "lstm"
        self.cell = (nn.LSTMCell if self.lstm else nn.GRUCell)(obs_dim, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out)) if deep_head else nn.Linear(hidden, out)
        if random_init:  # the target: orthogonal weights with a large gain so its outputs are not all near zero
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.orthogonal_(p, 2**0.5)
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x, reset):
        h = x.new_zeros(x.shape[1], self.hidden)
        c = torch.zeros_like(h)
        outs = []
        for xt, rt in zip(x, reset):
            m = (1.0 - rt).unsqueeze(-1)
            h, c = h * m, c * m
            if self.lstm:
                h, c = self.cell(xt, (h, c))
            else:
                h = self.cell(xt, h)
            outs.append(self.head(h))
        return torch.stack(outs)


def main():
    args, device, app = parse_args(Args, "ppo_rnd_rnn")
    seed_everything(args.seed)
    env = make_env(args, device)
    N, T, D, A, dev = env.num_envs, args.num_steps, env.obs_dim, env.act_dim, env.device
    batch = N * T
    mb = min(args.minibatch_size, batch)
    iters = args.total_timesteps // batch if args.total_timesteps else args.max_epochs
    want = args.rnd_seq_len or T
    L = max(d for d in range(1, min(want, T) + 1) if T % d == 0)  # window length: largest divisor of num_steps <= want
    if L != want:
        print(f"[ppo_rnd_rnn] rnd_seq_len {want} does not divide num_steps {T}: using {L}")
    n_chunks = T // L
    # seq: (T, N, ...) -> (L, n_chunks * N, ...): the rollout of every env becomes n_chunks windows of L steps, all
    # windows stacked along the batch axis. unseq undoes it: (L, n_chunks * N) -> (T, N).
    seq = lambda x: x.reshape(n_chunks, L, N, *x.shape[2:]).transpose(0, 1).reshape(L, n_chunks * N, *x.shape[2:])
    unseq = lambda x: x.reshape(L, n_chunks, N).transpose(0, 1).reshape(T, N)

    agent = Agent(D, A, args.hidden_units, args.activation).to(dev)
    target = SeqNet(D, args.rnd_hidden, args.rnd_dim, args.rnd_cell, random_init=True).to(dev)
    predictor = SeqNet(D, args.rnd_hidden, args.rnd_dim, args.rnd_cell, deep_head=True).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    rnd_opt = torch.optim.Adam(predictor.parameters(), lr=args.rnd_lr)
    obs_rms, val_rms = RunningMeanStd((D,), dev), RunningMeanStd((), dev)
    rnd_obs_rms, int_ret_rms = RunningMeanStd((D,), dev), RunningMeanStd((), dev)
    norm = lambda x: obs_rms.normalize(x, 10.0) if args.normalize_input else x
    rnd_norm = lambda x: rnd_obs_rms.normalize(x, 5.0)
    denorm_v = lambda v: val_rms.denormalize(v) if args.normalize_value else v
    norm_v = lambda v: val_rms.normalize(v) if args.normalize_value else v
    rnd_error = lambda x, reset: ((predictor(x, reset) - target(x, reset)) ** 2).mean(-1)  # (L, B)

    log, stats, lr = Logger("ppo_rnd_rnn", args, env), EpisodeStats(N, dev), args.learning_rate
    z = lambda *s: torch.zeros(T, N, *s, device=dev)
    b_obs, b_next, b_act, b_logp = z(D), z(D), z(A), z()
    b_rew, b_done, b_val, b_ival = z(), z(), z(), z()
    obs, step = env.reset(), 0
    int_return = torch.zeros(N, device=dev)

    for it in range(iters):
        if args.lr_schedule == "linear":
            lr = args.learning_rate * (1 - it / iters)
            opt.param_groups[0]["lr"] = lr
        for t in range(T):
            if args.normalize_input:
                obs_rms.update(obs)
            with torch.no_grad():
                mu, v, iv = agent(norm(obs))
                dist = Normal(mu, agent.logstd.exp())
                action = dist.sample()
            next_obs, rew, term, trunc = env.step(action.clamp(-args.clip_actions, args.clip_actions))
            v = denorm_v(v)
            r = rew * args.reward_scale + args.gamma * v * trunc.float()
            b_obs[t], b_next[t], b_act[t], b_logp[t] = obs, next_obs, action, dist.log_prob(action).sum(-1)
            b_rew[t], b_done[t], b_val[t], b_ival[t] = r, (term | trunc).float(), v, iv
            stats.update(rew, term | trunc)
            obs, step = next_obs, step + N

        # ---- intrinsic reward: error of the sequence predictor on the windows of next states
        rnd_obs_rms.update(b_next.reshape(-1, D))
        s_next, s_reset = seq(rnd_norm(b_next)), seq(b_done)  # next_obs[t] of a finished step already starts a new episode
        with torch.no_grad():
            r_int = unseq(rnd_error(s_next, s_reset))
            rets = []
            for t in range(T):
                int_return = int_return * args.int_gamma + r_int[t]
                rets.append(int_return.clone())
            int_ret_rms.update(torch.stack(rets))
            r_int = r_int / torch.sqrt(int_ret_rms.var + 1e-8)
            next_v, next_iv = agent(norm(obs))[1:]
        adv_ext = compute_gae(b_rew, b_val, b_done, denorm_v(next_v), args.gamma, args.gae_lambda)
        adv_int = compute_gae(r_int, b_ival, torch.zeros_like(b_done), next_iv, args.int_gamma, args.gae_lambda)  # non-episodic
        ret_ext, ret_int = adv_ext + b_val, adv_int + b_ival
        if args.normalize_value:
            val_rms.update(ret_ext)
        adv = args.ext_coef * adv_ext + args.int_coef * adv_int
        f_obs, f_act, f_logp, f_adv = norm(b_obs.reshape(-1, D)), b_act.reshape(-1, A), b_logp.reshape(-1), adv.reshape(-1)
        f_ret, f_val, f_ret_i, f_val_i = norm_v(ret_ext.reshape(-1)), norm_v(b_val.reshape(-1)), ret_int.reshape(-1), b_ival.reshape(-1)

        for _ in range(args.update_epochs):
            perm, kls = torch.randperm(batch, device=dev), []
            for s in range(0, batch, mb):
                i = perm[s:s + mb]
                mu, v, iv = agent(f_obs[i])
                dist = Normal(mu, agent.logstd.exp())
                logratio = dist.log_prob(f_act[i]).sum(-1) - f_logp[i]
                ratio = logratio.exp()
                kls.append(((ratio - 1) - logratio).mean().detach())
                a = f_adv[i]
                a = (a - a.mean()) / (a.std() + 1e-8) if args.norm_adv else a
                pg_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
                if args.clip_vloss:
                    v_clip = f_val[i] + (v - f_val[i]).clamp(-args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max((v - f_ret[i]) ** 2, (v_clip - f_ret[i]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((v - f_ret[i]) ** 2).mean()
                v_loss = v_loss + 0.5 * ((iv - f_ret_i[i]) ** 2).mean()
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

        # ---- predictor update on whole windows (a random subset of them, as in RND)
        n_seq = n_chunks * N
        for _ in range(args.update_epochs):
            perm = torch.randperm(n_seq, device=dev)
            for s in range(0, n_seq, args.rnd_batch_seqs):
                e = perm[s:s + args.rnd_batch_seqs]
                err = rnd_error(s_next[:, e], s_reset[:, e]).mean(0)  # (B,) mean over the window
                keep = (torch.rand_like(err) < args.update_proportion).float()
                rnd_loss = (err * keep).sum() / err.new_tensor(max(args.update_proportion * len(e), 1.0))
                rnd_opt.zero_grad()
                rnd_loss.backward()
                rnd_opt.step()

        log.log(step, **{"charts/sps": log.sps(step), "charts/lr": lr, "charts/approx_kl": kl,
                         "charts/mean_step_reward": stats.mean_step_reward(), "charts/ep_return": stats.mean_return(),
                         "rnd/intrinsic_reward": r_int.mean().item(), "rnd/loss": rnd_loss.item(),
                         "loss/policy": pg_loss.item(), "loss/value": v_loss.item()})
        print(f"it {it + 1}/{iters} step {step} sps {log.sps(step):.0f} step_rew {stats.mean_step_reward():.4f} "
              f"ep_ret {stats.mean_return():.2f} r_int {r_int.mean().item():.3f} rnd_loss {rnd_loss.item():.4f} kl {kl:.4f}")
        if args.save_interval and step % args.save_interval < batch:
            log.save_model(f"model_{step}", agent.state_dict())

    log.save_model("model_last", agent.state_dict(), obs_rms=(obs_rms.mean, obs_rms.var),
                   rnd=dict(predictor=predictor.state_dict(), target=target.state_dict()))
    write_result(args, stats, step)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
