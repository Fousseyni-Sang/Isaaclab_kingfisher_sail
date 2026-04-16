# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--task_name", type=str, default=None, help="directory name for the task")
parser.add_argument("--model_id", type=str, default=None, help="model id for loading and saving checkpoints")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--torch_deterministic", type=bool, default=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
parser.add_argument("--cuda", type=bool, default=True, help="if toggled, cuda will be enabled by default")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import os
import random
import time
from dataclasses import dataclass
import os
from datetime import datetime
import gymnasium as gym
import numpy as np
import torch
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab_tasks  # noqa
from omni.isaac.lab.utils.assets import retrieve_file_path

from omni.isaac.lab_tasks.utils import get_checkpoint_path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import torch.nn.functional as F
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from torch.utils.tensorboard import SummaryWriter

from omni.isaac.lab_tasks.utils.my_utils.clearnrl_agent import Agent, ExtractObsWrapper, RecordEpisodeStatisticsTorch, layer_init
def main():
    args_cli = parser.parse_args()

    spec_path = os.environ.get("LL_OUTPUT_DIR", None)
    temp = spec_path
    if spec_path==None and args_cli.model_id is not None:
        spec_path = f"outputs/ll/ll_model_{args_cli.model_id}" 
        os.environ["LL_OUTPUT_DIR"] = spec_path


    # specify directory for logging experiments
    if args_cli.task_name is not None:
        log_root_path = os.path.join("logs", "cleanrl", args_cli.task_name)
    else:
        log_root_path = os.path.join("logs", "cleanrl", f"kingfisher_sail_direct_hl" if args_cli.model_id is None 
                                     else f"kingfisher_sail_direct_ll_{args_cli.model_id}")
    log_root_path = os.path.abspath(log_root_path)
    # TRY NOT TO MODIFY: seeding
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.backends.cudnn.deterministic = args_cli.torch_deterministic

    device = torch.device(args_cli.device if torch.cuda.is_available() and args_cli.cuda else "cpu")

    # env setup
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, device=args_cli.device)
    envs = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    envs.unwrapped.is_Training = False
    # find checkpoint
    if args_cli.checkpoint is None:
        
        checkpoint_file = ".*"
        run_dir = ".*"
        print(f"join checkpoint file: {os.path.join(log_root_path, run_dir, checkpoint_file)}")
        # get path to previous checkpoint
        
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    eval_log_dir = os.path.join(log_root_path, "eval", log_dir)
    os.makedirs(eval_log_dir, exist_ok=True)
    summary_dir = os.path.join(eval_log_dir, "summary")
    writer = SummaryWriter(log_dir=summary_dir)

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args_cli).items()])),
    )

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
    agent.load(resume_path)
    print(f"[INFO]: Loaded model checkpoint from: {resume_path}")

    iteration = 0
    global_step = 0
    next_obs, _ = envs.reset()
    episodes_finished = 0

    while simulation_app.is_running() and episodes_finished < args_cli.num_epochs:
        
        # Annealing the rate if instructed to do so.
        action, logprob, _, value = agent.get_action_and_value(next_obs)
            
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, next_done, info = envs.step(action)
        
        for idx, d in enumerate(next_done):
            if d:
                episodic_return = info["r"][idx].item()
                print(f"[INFO]episode={episodes_finished}, global_step={global_step}, episodic_return={episodic_return}")
                episodes_finished += 1
                
                writer.add_scalar("evals/charts/episodic_return", episodic_return, global_step)
                writer.add_scalar("evals/charts/episodic_length", info["l"][idx], global_step)

                if "log" in info:
                    for log_key, log_value in info["log"].items():
                        # Safely extract the scalar value whether it's a Tensor or a standard float
                        val = log_value.item() if hasattr(log_value, "item") else log_value
                        # log_key already contains the prefix like "Episode_Reward/" or "Metrics/"
                        writer.add_scalar(f"log/{log_key}", val, global_step)
                        global_step += envs.num_envs
                        
        
    print(f"\n[INFO] Finished training after {iteration} iterations and {global_step} steps.")


    envs.close()
    

if __name__ == "__main__":
    main()
    simulation_app.close()