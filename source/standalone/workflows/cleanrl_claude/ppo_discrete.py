"""PPO with a discretized action space: every action dimension has its own number of levels (--bin_thrusters,
--bin_rudder, --bin_sail) and the policy has one categorical distribution per dimension (MultiDiscrete).
Same yaml / hyper-parameters as ppo.py.

pysaac source/standalone/workflows/cleanrl_claude/ppo_discrete.py --headless --num_envs 4096 --bin_rudder 5 --bin_sail 5 --bin_thrusters 2
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from common import (MultiCategorical, action_bins, PPOArgs, Logger, EpisodeStats, RunningMeanStd, adapt_lr, compute_gae, make_env, mlp, parse_args,
                    seed_everything, write_result)


@dataclass
class Args(PPOArgs):
    bin_thrusters: int = 3  # levels for each of the 2 thrusters (2 = -1/+1, 3 = -1/0/+1)
    bin_rudder: int = 5
    bin_sail: int = 5


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, bins, units, act):
        super().__init__()
        self.act_dim, self.bins, self.maxb = act_dim, list(bins), max(bins)
        self.torso = mlp([obs_dim, *units], act, last_act=True)
        self.logits = nn.Linear(units[-1], act_dim * self.maxb)
        self.value = nn.Linear(units[-1], 1)

    def forward(self, x):
        h = self.torso(x)
        return MultiCategorical(self.logits(h).view(-1, self.act_dim, self.maxb), self.bins), self.value(h).squeeze(-1)


def main():
    args, device, app = parse_args(Args, "ppo_discrete")
    seed_everything(args.seed)
    env = make_env(args, device, bins=action_bins(args), mode="multi")
    N, T, D, A = env.num_envs, args.num_steps, env.obs_dim, env.act_dim
    batch = N * T
    mb = min(args.minibatch_size, batch)
    iters = args.total_timesteps // batch if args.total_timesteps else args.max_epochs

    agent = Agent(D, A, action_bins(args), args.hidden_units, args.activation).to(env.device)
    opt = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    obs_rms, val_rms = RunningMeanStd((D,), env.device), RunningMeanStd((), env.device)
    norm = lambda x: obs_rms.normalize(x, 10.0) if args.normalize_input else x
    denorm_v = lambda v: val_rms.denormalize(v) if args.normalize_value else v
    norm_v = lambda v: val_rms.normalize(v) if args.normalize_value else v

    log, stats, lr = Logger("ppo_discrete", args, env), EpisodeStats(N, env.device), args.learning_rate
    z = lambda *s, dt=torch.float32: torch.zeros(T, N, *s, device=env.device, dtype=dt)
    b_obs, b_act, b_logp, b_rew, b_done, b_val = z(D), z(A, dt=torch.long), z(), z(), z(), z()
    obs, step = env.reset(), 0

    for it in range(iters):
        if args.lr_schedule == "linear":
            lr = args.learning_rate * (1 - it / iters)
            opt.param_groups[0]["lr"] = lr
        for t in range(T):
            if args.normalize_input:
                obs_rms.update(obs)
            with torch.no_grad():
                dist, v = agent(norm(obs))
                action = dist.sample()
            next_obs, rew, term, trunc = env.step(action)
            v = denorm_v(v)
            r = rew * args.reward_scale + args.gamma * v * trunc.float()
            b_obs[t], b_act[t], b_logp[t], b_rew[t], b_done[t], b_val[t] = \
                obs, action, dist.log_prob(action), r, (term | trunc).float(), v
            stats.update(rew, term | trunc)
            obs, step = next_obs, step + N

        with torch.no_grad():
            next_v = denorm_v(agent(norm(obs))[1])
        adv = compute_gae(b_rew, b_val, b_done, next_v, args.gamma, args.gae_lambda)
        ret = adv + b_val
        if args.normalize_value:
            val_rms.update(ret)
        f_obs, f_act, f_logp = norm(b_obs.reshape(-1, D)), b_act.reshape(-1, A), b_logp.reshape(-1)
        f_adv, f_ret, f_val = adv.reshape(-1), norm_v(ret.reshape(-1)), norm_v(b_val.reshape(-1))

        for _ in range(args.update_epochs):
            perm, kls = torch.randperm(batch, device=env.device), []
            for s in range(0, batch, mb):
                i = perm[s:s + mb]
                dist, v = agent(f_obs[i])
                logratio = dist.log_prob(f_act[i]) - f_logp[i]
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
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * dist.entropy().mean()
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
