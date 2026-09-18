# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Loader for the reward/environment parameters of KingfisherSailEnvCfg.

The YAML file is resolved in this order:
  1. the ``path`` argument passed explicitly to :func:`load_reward_cfg`
  2. the ``KINGFISHER_REWARD_CFG`` environment variable
  3. the default ``reward_cfg.yaml`` shipped next to this module

This lets a sweep script point many training runs at different reward
configurations (via the environment variable) without touching the code.
"""

from __future__ import annotations

import os

import yaml

_DEFAULT_CFG_PATH = os.path.join(os.path.dirname(__file__), "reward_cfg.yaml")


def load_reward_cfg(cfg, path: str | None = None) -> str | None:
    """Load reward/environment parameters from YAML onto ``cfg`` in place.

    Args:
        cfg: A ``KingfisherSailEnvCfg`` instance (or any object exposing the same attributes).
        path: Optional explicit path to a reward-config YAML file. If ``None``, the
            ``KINGFISHER_REWARD_CFG`` environment variable is used, falling back to the
            default file shipped next to this module.

    Returns:
        The path of the file that was loaded, or ``None`` if no file was found.

    Raises:
        ValueError: If a key in the YAML does not match any existing attribute on ``cfg``.
    """
    cfg_path = path or os.environ.get("KINGFISHER_REWARD_CFG", _DEFAULT_CFG_PATH)

    if not cfg_path or not os.path.isfile(cfg_path):
        print(f"[reward_cfg_loader] No reward config found at '{cfg_path}', keeping KingfisherSailEnvCfg defaults.")
        return None

    with open(cfg_path, "r") as f:
        sections = yaml.safe_load(f) or {}

    for section_name, params in sections.items():
        if not isinstance(params, dict):
            continue
        for key, value in params.items():
            if not hasattr(cfg, key):
                raise ValueError(
                    f"[reward_cfg_loader] '{key}' (section '{section_name}' of '{cfg_path}') does not match "
                    f"any attribute on {type(cfg).__name__}. Fix the YAML or add the attribute to the cfg class."
                )
            setattr(cfg, key, value)

    print(f"[reward_cfg_loader] Loaded reward configuration from '{cfg_path}'.")
    return cfg_path
