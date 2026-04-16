# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from datetime import datetime
from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab.utils.dict import print_dict
import os
# sac_isaaclab.py

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--buffer_size", type=int, default=20_000_000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--policy_frequency", type=int, default=2, help="the frequency of training policy (delayed)")
parser.add_argument("--target_network_frequency", type=int, default=1, help="the frequency of updates for the target nerworks")
parser.add_argument("--task_name", type=str, default=None, help="directory name for the task")

parser.add_argument("--learning_rate", type=float, default=5e-4, help="Entropy regularization coefficient")
parser.add_argument("--autotune", type=bool, default=True, help="automatic tuning of the entropy coefficient")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--actor_hdim", type=int, default=128, help="hidden dimension for the actor network")
parser.add_argument("--critic_hdim", type=int, default=128, help="hidden dimension for the critic network")
parser.add_argument("--horizon_length", type=int, default=20, help="number of steps to run in each environment per policy rollout")
parser.add_argument("--ll_horizon_length", type=int, default=10, help="number of steps to run in each environment per low-level policy rollout")
parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__)[:-3], help="the name of this experiment")
parser.add_argument("--torch_deterministic", type=bool, default=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
parser.add_argument("--cuda", type=bool, default=True, help="if toggled, cuda will be enabled by default")
parser.add_argument("--norm_adv", type=bool, default=True, help="Toggles advantages normalization")
parser.add_argument("--clip_vloss", type=bool, default=False, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
parser.add_argument("--track", type=bool, default=False, help="if toggled, this experiment will be tracked with Weights and Biases")
parser.add_argument("--wandb_project_name", type=str, default="cleanRL", help="the wandb's project name")
parser.add_argument("--wandb_entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--capture_video", type=bool, default=False,help="whether to capture videos of the agent performances (check out `videos` folder)")
parser.add_argument("--anneal_lr", type=bool, default=False, help="Toggle learning rate annealing for policy and value networks")
parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
parser.add_argument("--clip_coef", type=float, default=2.0, help="value function coefficient for PPO")
parser.add_argument("--target_kl", type=float, default=0.016, help="the target KL divergence threshold for PPO")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="the maximum norm for the gradient clipping")
parser.add_argument("--vf_coef", type=float, default=0.99, help="the clip coefficient for PPO")
parser.add_argument("--ent_coef", type=float, default=0.0, help="the entropy coefficient for PPO")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
parser.add_argument("--num_minibatches", type=int, default=2, help="the number of mini-batches")
parser.add_argument("--update_epochs", type=int, default=5, help="the K epochs to update the policy")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=10_000_000, help="Interval between video recordings (in steps).")
parser.add_argument("--reward_scaler", type=float, default=1.0, help="the scale factor applied to the reward during training")
parser.add_argument("--model_id", type=str, default=None, help="low level model to run")
parser.add_argument("--save_frequency", type=int, default=50, help="Interval between model saves")

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy
import os
import random
import time
from dataclasses import dataclass

from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab_tasks  # noqa
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from omni.isaac.lab_tasks.utils.my_utils.control_agent import get_control_agent
from omni.isaac.lab_tasks.utils.my_utils.ros2_Node import DynamicsRlAgentPublisher
from omni.isaac.lab_tasks.utils.my_utils.common import deterministic_split, load_ll_population_with_split, compute_act_dim, compute_obs_dim
from omni.isaac.lab_tasks.utils.my_utils.clearnrl_agent import RecordEpisodeStatisticsTorch, layer_init, Agent, ExtractObsWrapper

# TRY NOT TO MODIFY: seeding
random.seed(args_cli.seed)
np.random.seed(args_cli.seed)
torch.manual_seed(args_cli.seed)
torch.backends.cudnn.deterministic = args_cli.torch_deterministic

device = torch.device(args_cli.device if torch.cuda.is_available() and args_cli.cuda else "cpu")


if __name__ == "__main__":
    
    spec_path = os.environ.get("LL_OUTPUT_DIR", None)
    temp = spec_path
    if spec_path==None and args_cli.model_id is not None:
        spec_path = f"outputs/ll/ll_model_{args_cli.model_id}" 
        os.environ["LL_OUTPUT_DIR"] = spec_path

    args_cli.batch_size = int(args_cli.num_envs * args_cli.horizon_length)
    args_cli.minibatch_size = int(args_cli.batch_size // args_cli.num_minibatches)

    # specify directory for logging experiments
    if args_cli.task_name is not None:
        log_root_path = os.path.join("logs", "cleanrl", args_cli.task_name)
    else:
        log_root_path = os.path.join("logs", "cleanrl", f"kingfisher_sail_direct_hl" if args_cli.model_id is None else f"kingfisher_sail_direct_ll_{args_cli.model_id}")
    
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nn_dir = os.path.join(log_root_path, log_dir, "nn")
    summary_dir = os.path.join(log_root_path, log_dir, "summary")
    log_root_path = os.path.abspath(log_root_path)

    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    

    #run_name = os.path.join(log_root_path, log_dir)
    #os.makedirs(run_name, exist_ok=True)
    os.makedirs(nn_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    """if args_cli.track:
        import wandb

        wandb.init(
            project=args_cli.wandb_project_name,
            entity=args_cli.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )"""
    writer = SummaryWriter(summary_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args_cli).items()])),
    )

    

    # env setup
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, device=args_cli.device, use_fabric=not args_cli.disable_fabric)
    envs = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        envs = gym.wrappers.RecordVideo(envs, **video_kwargs)

    envs = ExtractObsWrapper(envs)
    envs = RecordEpisodeStatisticsTorch(envs, device, args_cli.num_envs)
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args_cli.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args_cli.horizon_length, args_cli.num_envs, envs.single_observation_space.shape[1]), dtype=torch.float).to(device)
    actions = torch.zeros((args_cli.horizon_length, args_cli.num_envs, envs.single_action_space.shape[1]), dtype=torch.float).to(device)
    logprobs = torch.zeros((args_cli.horizon_length, args_cli.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((args_cli.horizon_length, args_cli.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((args_cli.horizon_length, args_cli.num_envs), dtype=torch.float).to(device)
    values = torch.zeros((args_cli.horizon_length, args_cli.num_envs), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_done = torch.zeros(args_cli.num_envs, dtype=torch.float).to(device)

    iteration = 0
    for iteration in range(1, args_cli.num_epochs + 1):
        print(f"\n[INFO] Starting epoch {iteration}/{args_cli.num_epochs}.")
        # Annealing the rate if instructed to do so.
        if args_cli.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args_cli.num_epochs
            lrnow = frac * args_cli.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args_cli.horizon_length):
            global_step += args_cli.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                agent.eval()
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data. --> Cleanrl
            # IMPLEMENT THE HIERARHICAL LOOP HERE
            #reward_low_level = torch.zeros_like(rewards[step]).to(device)
            #for ll_step in range(args_cli.ll_horizon_length):
                
            next_obs, rewards[step], next_done, info = envs.step(action)
            #reward_low_level += reward

            if 0 <= step <= 2: 
                for idx, d in enumerate(next_done):
                    if d:
                        episodic_return = info["r"][idx].item()
                        print(f"global_step={global_step}, episodic_return={episodic_return}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
                        if "consecutive_successes" in info:  # ShadowHand and AllegroHand metric
                            writer.add_scalar(
                                "charts/consecutive_successes", info["consecutive_successes"].item(), global_step
                            )
                            
                        break

            #rewards[step] = reward_low_level
        
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args_cli.horizon_length)):
                if t == args_cli.horizon_length - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1].float()
                    nextvalues = values[t + 1]
                delta = rewards[t] + args_cli.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args_cli.gamma * args_cli.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1, envs.single_observation_space.shape[1]))
        
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, envs.single_action_space.shape[1]))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args_cli.update_epochs):
            b_inds = torch.randperm(args_cli.batch_size, device=device)
            for start in range(0, args_cli.batch_size, args_cli.minibatch_size):
                end = start + args_cli.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args_cli.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args_cli.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args_cli.clip_coef, 1 + args_cli.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args_cli.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args_cli.clip_coef,
                        args_cli.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args_cli.ent_coef * entropy_loss + v_loss * args_cli.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args_cli.max_grad_norm)
                optimizer.step()

            if args_cli.target_kl is not None and approx_kl > args_cli.target_kl:
                break

        if iteration % args_cli.save_frequency == 0:
            model_path = os.path.join(nn_dir, f"model_{iteration}.pth")
            agent.save(model_path)
            print(f"[INFO] Saved model checkpoint to {model_path} at epoch {iteration}.")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    writer.close()
    envs.close()
    simulation_app.close()