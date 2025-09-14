import gym
import math
import os
import random
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
import register_bipedal

# Call registration at the top, so it runs in every process (including Ray workers)
#register_bipedal.register_bipedal()

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
            "max_epochs": 1600,
            "save_best_after": 25,
            "save_frequency": 100,
            "gamma": 0.99,
            "tau": 0.95,
            "lr_schedule": "constant",
            "kl_threshold": 0.02,
            "score_to_win": 20000,   # stop when solved
            "max_epochs": 1500,
            "num_actors": 128,
            "horizon_length": 48,
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

config['params']['config']['env_name'] = "BipedalWalkerCtx-v0"
config['params']['config']['vecenv_type'] = "RAY"


# ----------------------
# Training Runner
# ----------------------
runner = Runner()
runner.load(config)
runner.run({"train": True})
