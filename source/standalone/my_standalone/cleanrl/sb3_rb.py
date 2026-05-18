# Copyright notice
#
# This file contains code adapted from stable-baselines3
# (https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py)
# licensed under the MIT License.
#
# Copyright (c) 2019-2023 Antonin Raffin, Ashley Hill, Anssi Kanervisto,
# Maximilian Ernestus, Rinu Boney, Pavan Goli, and other contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Dict, Union, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


__all__ = [
    "BaseBuffer",
    "RolloutBuffer",
    "ReplayBuffer",
    "RolloutBufferSamples",
    "ReplayBufferSamples",
]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

from enum import Enum

class GoalSelectionStrategy(Enum):
    """
    The strategies for selecting new goals when
    creating artificial transitions.
    """

    # Select a goal that was achieved
    # after the current step, in the same episode
    FUTURE = 0
    # Select the goal that was achieved
    # at the end of the episode
    FINAL = 1
    # Select a goal that was achieved in the episode
    EPISODE = 2


# For convenience
# that way, we can use string to select a strategy
KEY_TO_GOAL_STRATEGY = {
    "future": GoalSelectionStrategy.FUTURE,
    "final": GoalSelectionStrategy.FINAL,
    "episode": GoalSelectionStrategy.EPISODE,
}



def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_device(device: th.device | str = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples | RolloutBufferSamples:
        """
        :param batch_inds:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)



class HerReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        env: Any, # Expecting a VecEnv with GoalEnv properties
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        n_sampled_goal: int = 4,
        goal_selection_strategy: str = "future",
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.env = env
        self.n_envs = n_envs
        self.device = device
        self.pos = 0
        self.full = False
        
        # HER specific attributes
        self.n_sampled_goal = n_sampled_goal
        self.her_ratio = 1 - (1.0 / (self.n_sampled_goal + 1))
        
        # Core storage (Dict format internally)
        self.observations = {k: np.zeros((buffer_size, n_envs, *v.shape), dtype=v.dtype) 
                             for k, v in observation_space.spaces.items()}
        self.next_observations = {k: np.zeros((buffer_size, n_envs, *v.shape), dtype=v.dtype) 
                                  for k, v in observation_space.spaces.items()}
        
        self.actions = np.zeros((buffer_size, n_envs, action_space.shape[0]), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        
        # Episode tracking
        self.ep_start = np.zeros((buffer_size, n_envs), dtype=np.int64)
        self.ep_length = np.zeros((buffer_size, n_envs), dtype=np.int64)
        self._current_ep_start = np.zeros(n_envs, dtype=np.int64)

    def add(self, obs, next_obs, action, reward, done, infos):
        # Store current episode start for this transition
        self.ep_start[self.pos] = self._current_ep_start.copy()

        # Update Internal Storage
        for k in self.observations.keys():
            self.observations[k][self.pos] = np.array(obs[k])
            self.next_observations[k][self.pos] = np.array(next_obs[k])

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        # Handle episode termination
        for env_idx in range(self.n_envs):
            if done[env_idx]:
                self._finish_episode(env_idx)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _finish_episode(self, env_idx):
        start = self._current_ep_start[env_idx]
        end = self.pos
        length = (end - start) if end >= start else (self.buffer_size - start + end)
        
        # Mark all steps in this episode with their total length
        indices = np.arange(start, start + length) % self.buffer_size
        self.ep_length[indices, env_idx] = length
        self._current_ep_start[env_idx] = (self.pos + 1) % self.buffer_size

    def _flatten_obs(self, obs_dict: Dict[str, np.ndarray]) -> th.Tensor:
        """Concatenates Dict obs into a single tensor: [obs, achieved, desired]"""
        # Order matters! Usually: observation, achieved_goal, desired_goal
        list_obs = [obs_dict["observation"], obs_dict["achieved_goal"], obs_dict["desired_goal"]]
        # Flatten and convert to torch
        flat = np.concatenate([o.reshape(o.shape[0], -1) for o in list_obs], axis=1)
        return th.as_tensor(flat, device=self.device)

    def sample(self, batch_size: int):
        # Valid indices are those where an episode has been completed
        upper_bound = self.buffer_size if self.full else self.pos
        is_valid = self.ep_length[:upper_bound] > 0
        valid_indices = np.flatnonzero(is_valid)
        
        sampled_indices = np.random.choice(valid_indices, size=batch_size, replace=True)
        batch_inds, env_inds = np.unravel_index(sampled_indices, (upper_bound, self.n_envs))

        # Split batch
        nb_virtual = int(self.her_ratio * batch_size)
        
        # 1. Process Real Samples
        r_batch, r_env = batch_inds[nb_virtual:], env_inds[nb_virtual:]
        real_obs = {k: v[r_batch, r_env] for k, v in self.observations.items()}
        real_next_obs = {k: v[r_batch, r_env] for k, v in self.next_observations.items()}
        real_rewards = self.rewards[r_batch, r_env]

        # 2. Process Virtual Samples (HER)
        v_batch, v_env = batch_inds[:nb_virtual], env_inds[:nb_virtual]
        virt_obs = {k: v[v_batch, v_env] for k, v in self.observations.items()}
        virt_next_obs = {k: v[v_batch, v_env] for k, v in self.next_observations.items()}
        
        # Sample 'future' goals
        starts = self.ep_start[v_batch, v_env]
        lengths = self.ep_length[v_batch, v_env]
        current_rel_step = (v_batch - starts) % self.buffer_size
        
        # Random step between [current, end of episode]
        offset = np.random.randint(current_rel_step, lengths)
        goal_indices = (starts + offset) % self.buffer_size
        new_goals = self.next_observations["achieved_goal"][goal_indices, v_env]
        
        virt_obs["desired_goal"] = new_goals
        virt_next_obs["desired_goal"] = new_goals
        
        # Recompute rewards via the environment
        virt_rewards = self.env.env_method(
            "compute_reward", virt_next_obs["achieved_goal"], virt_obs["desired_goal"], [{}], indices=[0]
        )[0]

        # 3. Concatenate and Flatten
        final_obs = {k: np.concatenate([virt_obs[k], real_obs[k]]) for k in virt_obs.keys()}
        final_next_obs = {k: np.concatenate([virt_next_obs[k], real_next_obs[k]]) for k in virt_next_obs.keys()}
        
        return ReplayBufferSamples(
            observations=self._flatten_obs(final_obs),
            actions=th.as_tensor(np.concatenate([self.actions[v_batch, v_env], self.actions[r_batch, r_env]]), device=self.device),
            next_observations=self._flatten_obs(final_next_obs),
            dones=th.as_tensor(np.concatenate([self.dones[v_batch, v_env], self.dones[r_batch, r_env]]), device=self.device).reshape(-1, 1),
            rewards=th.as_tensor(np.concatenate([virt_rewards, real_rewards]), device=self.device).reshape(-1, 1)
        )
    
