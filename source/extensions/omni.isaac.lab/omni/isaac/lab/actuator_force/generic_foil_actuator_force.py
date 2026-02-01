from dataclasses import dataclass, field
from typing import List, Tuple
import torch
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.physics.foil_dynamics import FoilDynamics
from .foil_actuator_force import FoilActuatorCfg, FoilActuator

@configclass
class GenFoilActuatorCfg:
    foils: List[FoilActuatorCfg] = field(default_factory=list)
    
    def __post_init__(self):
        self.num_foils = len(self.foils)
        self.foil_types = [foil_cfg.foil_type for foil_cfg in self.foils]

class GenFoilActuator:
    def __init__(self, num_envs: int, device, dt: float,
                 cfg: GenFoilActuatorCfg,
                 dynamics_list: List[FoilDynamics]):
        """
        dynamics_list: list of FoilDynamics, one per foil (sail, keel, rudder, ...)
        """
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.cfg = cfg

        if cfg.num_foils != len(dynamics_list):
            raise ValueError(f"cfg.num_foils={cfg.num_foils} but got {len(dynamics_list)} FoilDynamics objects")

        self.num_foils = cfg.num_foils

        # For now: 1D command per foil (angle command)
        self._foil_cmd_dims = [1 for _ in range(self.num_foils)]
        if "keel" in self.cfg.foil_types:
            self._foil_cmd_dims.pop(-1) # keel actuator is not controled by the agent thus has no command
        self.total_cmd_dim = sum(self._foil_cmd_dims) 

        # Flat command tensors
        self._current_cmds = torch.zeros(
            (self.num_envs, self.total_cmd_dim), dtype=torch.float32, device=self.device
        )
        self._target_cmds = torch.zeros_like(self._current_cmds)

        # Per-foil actuators
        self.foil_actuators: List[FoilActuator] = []
        for foil_cfg, dyn in zip(self.cfg.foils, dynamics_list):
            self.foil_actuators.append(FoilActuator(num_envs, dyn, dt, foil_cfg))

        # Outputs: (num_envs, num_foils, 6)
        self.foil_forces_torques = torch.zeros(
            (self.num_envs, self.num_foils, 6), dtype=torch.float32, device=self.device
        )

        # Command offsets
        self._cmd_offsets = []
        offset = 0
        for dim in self._foil_cmd_dims:
            self._cmd_offsets.append(offset)
            offset += dim

        self.reset()

    def set_target_cmd(self, commands: torch.Tensor):
        """
        commands: (num_envs, total_cmd_dim)
        """
        if commands.shape != self._target_cmds.shape:
            raise ValueError(
                f"Expected commands of shape {self._target_cmds.shape}, got {commands.shape}"
            )
        self._target_cmds = commands.to(self.device)

    def update_joint_cmds(self, current_joint_positions: List[torch.Tensor]):
        """
        current_joint_positions: list of tensors, one per foil, each (num_envs, 1)
        """
        for i, foil_act in enumerate(self.foil_actuators):
            # Since foil_actuators might contain "keel" actuator so make sure the index doesn't exceed 
            # Even when the update is called in case of "keel", inside, the function doesn't update the joint command anyway
            
            offset = self._cmd_offsets[min(i, self.total_cmd_dim-1)] 
            cmd_slice = self._target_cmds[:, offset:offset+1]
            foil_act.update_joint_cmd(current_joint_positions[i], cmd_slice)

    def update_forces(self, robot_heading_w, robot_lin_vel_b):
        """
        Update forces for all foils and store (num_envs, num_foils, 6)
        """
        for i, foil_act in enumerate(self.foil_actuators):
            foil_act.update_forces(robot_heading_w, robot_lin_vel_b)
            self.foil_forces_torques[:, i, :] = foil_act.get_forces_and_torques()

        return self.foil_forces_torques

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._current_cmds[env_ids, :] = 0.0
        self._target_cmds[env_ids, :] = 0.0
        self.foil_forces_torques[env_ids, :, :] = 0.0
        for foil_act in self.foil_actuators:
            foil_act.reset(env_ids)

