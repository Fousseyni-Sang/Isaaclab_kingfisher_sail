import os
import gymnasium as gym
import numpy as np
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner
import argparse
import torch
from rl_games.algos_torch import model_builder
from wrappers import ContextWrapper
# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_episode", type=int, default=2, help="number of episodes for evaluation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")

args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True


# Training config for rl_games PPO
config = {
    "params": {
        "seed": 42,
        "algo": {
            "name": "a2c_continuous"
        },
        "model": {
            "name": "continuous_a2c_logstd"
        },
        "network": {
            "name": "actor_critic",
            "separate": False,
            "space": {
                "continuous": {
                    "mu_activation": "None",
                    "sigma_activation": "None",
                    "mu_init": {"name": "default"},
                    "sigma_init": {"name": "const_initializer", "val": 0},
                    "fixed_sigma": True,
                }
            },
            "mlp": {
                "units": [128, 128, 128],
                "activation": "tanh",
                "initializer": {"name": "default"}
            }
        },
        "config": {
            "name": "bipedal_walker",
            "env_name": "BipedalWalkerCtx-v0",
            "device": 'cuda:0',
            "device_name": 'cuda:0',
            "multi_gpu": False,
            "ppo": True,
            "reward_shaper":{'scale_value': 0.1},
            "vecenv_type": "RAY",   
            "normalize_input": True,
            "normalize_value": True,
            "max_epochs": 600,
            "save_best_after": 25,
            "save_frequency": 50,
            "gamma": 0.99,
            "tau": 0.95,
            'player': {'render': True},
            "lr_schedule": "constant",
            "kl_threshold": 0.02,
            "score_to_win": 1000,   # stop when solved
            "max_epochs": 2000,
            "num_actors": 2,
            "horizon_length": 16,
            "minibatch_size": 2048,
            "mini_epochs": 4,
            "e_clip": 0.2, #0.2
            "clip_value": True,
            "clip_param": 0.2,
            "value_loss_coef": 2.0,
            "entropy_coef": 0.0,
            "learning_rate": 3e-4,
            "normalize_advantage": True,
            "critic_coef": 2,
            "seq_length": 4,
            "bounds_loss_coef": 0.0001,
            "grad_norm": 1.0,
            "truncate_grads": True,
           
        }
    }
}


def evaluate(agent_path, episodes=5, render=False, record_video=False):
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array" if args_cli.video else None)
    # optional video recording
    
    if args_cli.video:
        print(f"========== VIDEO: {args_cli.video}")
        video_kwargs = {
            "video_folder": os.path.join("bipedal_walker_04-13-53-32", "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        #print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = ContextWrapper(env, context_dim=2, resample_every=50, low=-1.0, high=1.0, device="cuda:0")

    config['params']['config']['player']['render'] = True if render else False
    config['params']['config']['player']['games_num'] = 1
    runner = Runner()
    runner.load(config)
    agent = runner.create_player()
    agent.restore(agent_path)

    # ---- Evaluation Loop ----
    episode_cntr = 0
    max_episode_length = 1600
    num_episodes = args_cli.num_episode
    max_episod_length = max_episode_length

    lin_vel_x_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.device)
    acord_prediction_logs = torch.zeros((max_episod_length, args_cli.num_envs, 3), device=env.device)
    vel_context_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.device)

    angle_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.device)
    angle_context_logs = torch.zeros((max_episod_length, args_cli.num_envs), device=env.device)

    lin_vel_x_list = []
    acord_prediction_list = []
    vel_context_list = []
    episode_lengths_list = []
    angle_list = []
    angle_context_list = []

    """agent.env = env
    agent.run()"""
    rewards = []
    
    for ep in range(args_cli.num_episode):
        obs, info = env.reset()
        done = False
        total_r = 0
        it = 0
        while not done:
            #obs_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            #print(f"obs_v: {obs_v.device}")
            with torch.no_grad():
                obs = agent.obs_to_torch(obs)
                act = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, r, done, _, _ = env.step(act.cpu().numpy())
            it+=1

            vel_context = obs[-2]
            angle_context = obs[-1]

            total_r += r
            if render:
                env.render()

        rewards.append(total_r)
        print(f"Episode {ep+1}: {total_r:.2f}")
        print(f"total_iter: {it}")

        lin_vel_x_logs[it] = obs[2].clone().float()
        vel_context_logs[it] = vel_context.clone().float()
        angle_logs[it] = obs[0].clone().float()
        angle_context_logs[it] = angle_context.clone().float()

        episode_lengths_list.append(it)

        lin_vel_x_list.append(lin_vel_x_logs[:-1].clone())
        vel_context_list.append(vel_context_logs[:-1].clone())
        angle_list.append(angle_logs[:-1].clone())
        angle_context_list.append(angle_context_logs[:-1].clone())

    env.close()
    print(f"Average reward over {episodes} episodes: {np.mean(rewards):.2f}")

    return lin_vel_x_list, vel_context_list, episode_lengths_list, angle_list, angle_context_list

# Example usage:
if __name__ == "__main__":
    # run the main execution
    lin_vel_x_list, vel_context_list, episode_lengths_list, angle_list, angle_context_list = \
    evaluate("runs/bipedal_walker_04-13-53-32/nn/last_bipedal_walker_ep_300_rew_269.9858.pth",
         episodes=10, record_video=True, render=True)

    import pandas as pd
    import os
    from datetime import datetime

    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = os.path.join("runs/bip_rlgames", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving logs to: {output_dir}")

    # --- Save lin_vel_x, vel_context ---
    vel_context = []
    for ep_idx, (lin_vel_x, context) in enumerate(zip(lin_vel_x_list, vel_context_list)):  # (T, N, 2)
        for t in range(lin_vel_x.shape[0]):
            for env_id in range(lin_vel_x.shape[1]):
                vel_context.append({
                    "episode": ep_idx,
                    "time_step": t,
                    "env_id": env_id,
                    "lin_vel_x": lin_vel_x[t, env_id].item(),
                    "vel_context": context[t, env_id].item(),
                    "angle": angle_list[ep_idx][t, env_id].item(),
                    "angle_context": angle_context_list[ep_idx][t, env_id].item()
                })
    pd.DataFrame(vel_context).to_csv(os.path.join(output_dir, "acord.csv"), index=False)
    
     # --- Save episode length ---
    episode_length_records = []
    for ep_idx, length in enumerate(episode_lengths_list):

        episode_length_records.append({
            "episode":ep_idx, 
            "ep_length": length
        })
    pd.DataFrame(episode_length_records).to_csv(os.path.join(output_dir, "ep_length.csv"), index=False)

