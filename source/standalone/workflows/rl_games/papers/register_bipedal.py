import gym
import numpy as np
from rl_games.common import env_configurations
from wrappers import ContextWrapper  # wherever you define it

def make_bipedal_ctx_env(**kwargs):
    env = gym.make("BipedalWalker-v3")
    env = ContextWrapper(env, context_dim=2, resample_every=50, low=-1.0, high=1.0, device="cuda:0")
    return env

# Register in rl_games
env_configurations.configurations['BipedalWalkerCtx-v0'] = {
    'env_creator': make_bipedal_ctx_env,
    'vecenv_type': 'RAY'
}
