# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from omni.isaac.lab.app import AppLauncher

# sac_isaaclab.py

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--total_timesteps", type=int, default=1_000_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--buffer_size", type=int, default=1e6)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--policy_frequency", type=int, default=2, help="the frequency of training policy (delayed)")
parser.add_argument("--target_network_frequency", type=int, default=1, help="the frequency of updates for the target nerworks")
parser.add_argument("--alpha", type=float, default=0.2, help="Entropy regularization coefficient")
parser.add_argument("--autotune", type=bool, default=True, help="automatic tuning of the entropy coefficient")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import os
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab_tasks  # noqa
import gymnasium as gym
from buffers import ReplayBuffer

# SAC Networks
class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))
        
def to_tensor_batch(batch_column):
    return torch.stack([
        item["policy"] if isinstance(item, dict) else item
        for item in batch_column
    ])

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, action_scale):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
        self.LOG_STD_MIN = -5
        self.LOG_STD_MAX = 2
        self.action_scale = action_scale

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6)
        return torch.clamp(action * self.action_scale, -1.0, 1.0), log_prob.sum(dim=-1, keepdim=True)


def main():
    

    # Prepare IsaacLab environment
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs, device=args.device, use_fabric=not args.disable_fabric)
    env = gym.make(args.task, cfg=env_cfg)
    obs_space = env.observation_space
    act_space = env.action_space

    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]
    action_scale = 1 #torch.tensor((act_space.high - act_space.low) / 2.0, device=args.device)
    
    #print(f"action scale: {action_scale}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Networks
    actor = Actor(obs_dim, act_dim, action_scale).to(args.device)
    qf1 = SoftQNetwork(obs_dim, act_dim).to(args.device)
    qf2 = SoftQNetwork(obs_dim, act_dim).to(args.device)
    qf1_target = SoftQNetwork(obs_dim, act_dim).to(args.device)
    qf2_target = SoftQNetwork(obs_dim, act_dim).to(args.device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=1e-3)
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)

    # Alpha tuning
    if args.autotune:
       log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
       alpha_optimizer = optim.Adam([log_alpha], lr=1e-3)
       target_entropy = -act_dim
    else:
       alpha = args.alpha

    # Replay Buffer (simple list for brevity, use CleanRL’s ReplayBuffer for prod)
    #buffer = []
    
    rb = ReplayBuffer(
        args.buffer_size,
        obs_space,
        act_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    # Logging
    run_name = f"{args.task}__sac__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    # Start loop
    """obs, _ = env.reset(seed=args.seed)
    obs = torch.tensor(obs, dtype=torch.float32, device=args.device)"""
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["policy"]
    global_step = 0

    while simulation_app.is_running() and global_step < args.total_timesteps:
        with torch.inference_mode():
            if global_step < 5000:
                #print(act_space.sample(), torch.tensor(act_space.sample(), device=args.device).shape)
                action = torch.tensor(act_space.sample(), device=args.device) #.unsqueeze(0)#.repeat(args.num_envs, 1)
            else:
                action, _, _ = actor.get_action(obs)
        
        
        next_obs, reward, done, trunc, info = env.step(action)
        #print(f"next_obs: {next_obs}")
        if isinstance(next_obs, dict):
           next_obs = next_obs["policy"]
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=args.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=args.device)
        done = torch.tensor(done, dtype=torch.float32, device=args.device)
        # Ensure obs, next_obs are tensors
        if isinstance(obs, tuple):
           obs = obs[0]
        if isinstance(next_obs, tuple):
           next_obs = next_obs[0]
        if isinstance(reward, tuple):
           reward = reward[0]
        if isinstance(done, tuple):
           done = done[0]
        
        """buffer.append((obs, action, reward, next_obs, done))
        if len(buffer) > 1_000_000:
            buffer.pop(0)"""
            
        rb.add(obs, next_obs, actions, rewards, done)

        obs = next_obs
        global_step += args.num_envs

        if global_step >= 5000:
            # Sample minibatch
            if isinstance(next_obs, dict):
               next_obs = next_obs["policy"]
               
            if isinstance(obs, dict):
               obs = obs["policy"]
               
            data = rb.sample(args.batch_size)
            #idx = np.random.randint(0, len(buffer), size=256)
            #batch = [buffer[i] for i in idx]
            #print(f"obs: {obs}\n, action: {action}\n, rew: {reward}\n, next_obs: {next_obs}\n, done: {done}")
            
            #b_obs, b_action, b_reward, b_next_obs, b_done = map(to_tensor_batch, zip(*batch))

            with torch.no_grad():
                next_action, next_log_prob = actor.get_action(data.next_observations)
                  
                qf1_target_val = qf1_target(data.next_observations, next_action)
                qf2_target_val = qf2_target(data.next_observations, next_action)
                q_target = torch.min(qf1_target_val, qf2_target_val) - log_alpha.exp() * next_log_prob
                target = data.rewards + 0.99 * (1 - data.dones.flatten()) * q_target.view(-1)

            qf1_loss = F.mse_loss(qf1(data.observations, data.actions).view(-1), target)
            qf2_loss = F.mse_loss(qf2(data.observations, data.actions).view(-1), target)
            q_optimizer.zero_grad()
            (qf1_loss + qf2_loss).backward()
            q_optimizer.step()

            # Policy update
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    
		    pi, log_pi = actor.get_action(b_obs) 
		    min_qf_pi = torch.min(qf1(b_obs, pi), qf2(b_obs, pi))
		    actor_loss = (log_alpha.exp() * log_pi - min_qf_pi).mean()
		    actor_optimizer.zero_grad()
		    actor_loss.backward()
		    actor_optimizer.step()

            # Alpha update
            alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            # Target soft update
            tau = 0.005
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            writer.add_scalar("loss/qf1", qf1_loss.item(), global_step)
            writer.add_scalar("loss/qf2", qf2_loss.item(), global_step)
            writer.add_scalar("loss/actor", actor_loss.item(), global_step)
            writer.add_scalar("loss/alpha", alpha_loss.item(), global_step)

    env.close()
    writer.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
