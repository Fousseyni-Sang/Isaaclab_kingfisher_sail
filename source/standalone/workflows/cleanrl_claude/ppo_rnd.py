"""PPO + Random Network Distillation (Burda et al. 2018) for Isaac Lab. Same yaml / hyper-parameters as ppo.py.

The intrinsic reward is the prediction error of a trained predictor network against a frozen random target
network, computed on the next observation. The agent has two value heads (extrinsic, episodic, and intrinsic,
non-episodic) and the advantage is  ext_coef * A_ext + int_coef * A_int.

pysaac source/standalone/workflows/cleanrl_claude/ppo_rnd.py --headless --num_envs 4096
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Normal

from common import (PPOArgs, Logger, EpisodeStats, RunningMeanStd, adapt_lr, compute_gae, make_env, mlp, parse_args,
                    seed_everything, write_result)


@dataclass
class Args(PPOArgs):
    int_coef: float = 1.0
    ext_coef: float = 2.0
    int_gamma: float = 0.99
    rnd_lr: float = 1e-4
    rnd_dim: int = 64
    rnd_hidden: int = 128
    update_proportion: float = 0.25  # fraction of each mini-batch used to train the predictor


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, units, act):
        super().__init__()
        self.torso = mlp([obs_dim, *units], act, last_act=True)
        self.mu, self.v_ext, self.v_int = nn.Linear(units[-1], act_dim), nn.Linear(units[-1], 1), nn.Linear(units[-1], 1)
        self.logstd = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        h = self.torso(x)
        return self.mu(h), self.v_ext(h).squeeze(-1), self.v_int(h).squeeze(-1)


class RND(nn.Module):
    def __init__(self, obs_dim, hidden, out):
        super().__init__()
        self.target = mlp([obs_dim, hidden, hidden, out], "relu")
        self.predictor = mlp([obs_dim, hidden, hidden, hidden, out], "relu")
        for p in self.target.parameters():
            p.requires_grad_(False)

    def error(self, x):  # per-sample prediction error, (B,)
        return ((self.predictor(x) - self.target(x)) ** 2).mean(-1)


def main():
    args, device, app = parse_args(Args, "ppo_rnd")
    seed_everything(args.seed)
    env = make_env(args, device)
    N, T, D, A, dev = env.num_envs, args.num_steps, env.obs_dim, env.act_dim, env.device
    batch = N * T
    mb = min(args.minibatch_size, batch)
    iters = args.total_timesteps // batch if args.total_timesteps else args.max_epochs

    agent = Agent(D, A, args.hidden_units, args.activation).to(dev)
    rnd = RND(D, args.rnd_hidden, args.rnd_dim).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    rnd_opt = torch.optim.Adam(rnd.predictor.parameters(), lr=args.rnd_lr)
    obs_rms, val_rms = RunningMeanStd((D,), dev), RunningMeanStd((), dev)
    rnd_obs_rms, int_ret_rms = RunningMeanStd((D,), dev), RunningMeanStd((), dev)
    norm = lambda x: obs_rms.normalize(x, 10.0) if args.normalize_input else x
    rnd_norm = lambda x: rnd_obs_rms.normalize(x, 5.0)
    denorm_v = lambda v: val_rms.denormalize(v) if args.normalize_value else v
    norm_v = lambda v: val_rms.normalize(v) if args.normalize_value else v

    log, stats, lr = Logger("ppo_rnd", args, env), EpisodeStats(N, dev), args.learning_rate
    z = lambda *s: torch.zeros(T, N, *s, device=dev)
    b_obs, b_next, b_act, b_logp = z(D), z(D), z(A), z()
    b_rew, b_done, b_val, b_ival = z(), z(), z(), z()
    obs, step = env.reset(), 0
    int_return = torch.zeros(N, device=dev)  # discounted intrinsic return (for reward normalisation)

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

        # ---- intrinsic reward = predictor error on next_obs, scaled by the std of its discounted return
        rnd_obs_rms.update(b_next.reshape(-1, D))
        with torch.no_grad():
            r_int = rnd.error(rnd_norm(b_next.reshape(-1, D))).view(T, N)
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
        f_rnd_obs = rnd_norm(b_next.reshape(-1, D))

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

                err = rnd.error(f_rnd_obs[i])  # predictor update on a random subset of the mini-batch
                rnd_loss = (err * (torch.rand_like(err) < args.update_proportion).float()).sum() / err.new_tensor(
                    max(args.update_proportion * len(i), 1.0))
                rnd_opt.zero_grad()
                rnd_loss.backward()
                rnd_opt.step()
            kl = torch.stack(kls).mean().item()
            if args.lr_schedule == "adaptive":
                lr = adapt_lr(lr, kl, args.kl_threshold)
                opt.param_groups[0]["lr"] = lr

        log.log(step, **{"charts/sps": log.sps(step), "charts/lr": lr, "charts/approx_kl": kl,
                         "charts/mean_step_reward": stats.mean_step_reward(), "charts/ep_return": stats.mean_return(),
                         "rnd/intrinsic_reward": r_int.mean().item(), "rnd/loss": rnd_loss.item(),
                         "loss/policy": pg_loss.item(), "loss/value": v_loss.item()})
        print(f"it {it + 1}/{iters} step {step} sps {log.sps(step):.0f} step_rew {stats.mean_step_reward():.4f} "
              f"ep_ret {stats.mean_return():.2f} r_int {r_int.mean().item():.3f} lr {lr:.2e} kl {kl:.4f}")
        if args.save_interval and step % args.save_interval < batch:
            log.save_model(f"model_{step}", agent.state_dict())

    log.save_model("model_last", agent.state_dict(), obs_rms=(obs_rms.mean, obs_rms.var), rnd=rnd.state_dict())
    write_result(args, stats, step)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
