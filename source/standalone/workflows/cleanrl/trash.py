import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import os
import time
import random
import yaml
from dataclasses import dataclass
from typing import Any
from torch.utils.tensorboard import SummaryWriter

from isaaclab.envs import IsaacEnv  # replace with your custom task

# --------- CONFIGURATION ---------
@dataclass
class SACConfig:
    env_id: str
    total_timesteps: int
    learning_rate: float
    buffer_size: int
    batch_size: int
    gamma: float
    tau: float
    train_freq: int
    gradient_steps: int
    start_training_timesteps: int
    policy_hidden_dim: int
    alpha: float
    eval_every: int
    checkpoint_path: str
    device: str

def load_config(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return SACConfig(**data)

# --------- MODELS ---------
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.net = MLP(obs_dim, 2 * act_dim, hidden_dim)

    def forward(self, x):
        mu_logstd = self.net(x)
        mu, logstd = mu_logstd.chunk(2, dim=-1)
        logstd = torch.clamp(logstd, -20, 2)
        std = torch.exp(logstd)
        dist = torch.distributions.Normal(mu, std)
        return dist

    def sample(self, x):
        dist = self.forward(x)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return torch.tanh(action), log_prob

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden_dim)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden_dim)

    def forward(self, x, a):
        xu = torch.cat([x, a], dim=-1)
        return self.q1(xu), self.q2(xu)

# --------- REPLAY BUFFER ---------
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = torch.zeros((size, obs_dim), device=device)
        self.next_obs_buf = torch.zeros((size, obs_dim), device=device)
        self.acts_buf = torch.zeros((size, act_dim), device=device)
        self.rews_buf = torch.zeros((size,), device=device)
        self.done_buf = torch.zeros((size,), device=device)
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.obs_buf.device)
        return (self.obs_buf[idx], self.acts_buf[idx], self.rews_buf[idx],
                self.next_obs_buf[idx], self.done_buf[idx])

# --------- TRAINING LOOP ---------
def train(cfg: SACConfig):
    env = IsaacEnv(cfg.env_id, device=cfg.device)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = Actor(obs_dim, act_dim, cfg.policy_hidden_dim).to(cfg.device)
    critic = Critic(obs_dim, act_dim, cfg.policy_hidden_dim).to(cfg.device)
    critic_target = Critic(obs_dim, act_dim, cfg.policy_hidden_dim).to(cfg.device)
    critic_target.load_state_dict(critic.state_dict())

    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg.learning_rate)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=cfg.learning_rate)

    rb = ReplayBuffer(obs_dim, act_dim, cfg.buffer_size, cfg.device)
    writer = SummaryWriter()

    obs = env.reset()
    global_step = 0

    while global_step < cfg.total_timesteps:
        if global_step < cfg.start_training_timesteps:
            action = torch.tensor(env.action_space.sample(), device=cfg.device)
        else:
            with torch.no_grad():
                action, _ = actor.sample(obs.unsqueeze(0))
                action = action.squeeze(0)

        next_obs, reward, done, info = env.step(action)
        rb.add(obs, action, reward, next_obs, done)

        obs = next_obs if not done else env.reset()
        global_step += 1

        if global_step >= cfg.start_training_timesteps:
            for _ in range(cfg.gradient_steps):
                o, a, r, no, d = rb.sample(cfg.batch_size)

                with torch.no_grad():
                    next_a, next_logp = actor.sample(no)
                    target_q1, target_q2 = critic_target(no, next_a)
                    target_q = torch.min(target_q1, target_q2) - cfg.alpha * next_logp.unsqueeze(-1)
                    target = r.unsqueeze(-1) + (1 - d.unsqueeze(-1)) * cfg.gamma * target_q

                current_q1, current_q2 = critic(o, a)
                critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()

                new_a, logp = actor.sample(o)
                q1_pi, q2_pi = critic(o, new_a)
                actor_loss = (cfg.alpha * logp - torch.min(q1_pi, q2_pi)).mean()
                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()

                # Polyak update
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

        # Logging & checkpoint
        if global_step % cfg.eval_every == 0:
            avg_reward = evaluate(env, actor, episodes=5, device=cfg.device)
            writer.add_scalar("eval/avg_reward", avg_reward, global_step)

            ckpt = {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "step": global_step
            }
            os.makedirs(cfg.checkpoint_path, exist_ok=True)
            torch.save(ckpt, os.path.join(cfg.checkpoint_path, f"sac_ckpt_{global_step}.pt"))

    writer.close()

def evaluate(env, actor, episodes=5, device="cuda"):
    total_reward = 0.0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action, _ = actor.sample(obs.unsqueeze(0))
                obs, reward, done, _ = env.step(action.squeeze(0))
                total_reward += reward.item()
    return total_reward / episodes


if __name__ == "__main__":
    cfg = load_config("cfg.yaml")
    train(cfg)
