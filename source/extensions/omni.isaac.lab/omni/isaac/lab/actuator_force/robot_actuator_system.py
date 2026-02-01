from omni.isaac.lab.actuator_force.generic_actuator_force import GenPropellerActuatorCfg, GenPropellerActuator
from omni.isaac.lab.actuator_force.generic_foil_actuator_force import GenFoilActuatorCfg, GenFoilActuator
import torch

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class RobotActuatorSystemCfg:
    # Thruster subsystem
    thruster_cfg: Optional[GenPropellerActuatorCfg] = None

    # Foil subsystem
    foil_cfg: Optional[GenFoilActuatorCfg] = None
    foil_dynamics: Optional[List] = None       # list of FoilDynamics

    # Optional: future actuators (winches, flaps, etc.)
    # winch_cfg: ...
    # winch_dynamics: ...

    # Sanity check
    def validate(self):
        
        if self.foil_cfg is not None:
            if self.foil_dynamics is None:
                raise ValueError("foil_cfg provided but foil_dynamics missing")
            if len(self.foil_cfg.foils) != len(self.foil_dynamics):
                raise ValueError("foil_cfg.foils and foil_dynamics length mismatch")


class RobotActuatorSystem:
    def __init__(self, num_envs, device, dt, cfg: RobotActuatorSystemCfg):
        cfg.validate()

        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.cfg = cfg

        # ------------------------------------------------------------
        # Instantiate thruster subsystem
        # ------------------------------------------------------------
        if cfg.thruster_cfg is not None:
            self.thruster_actuator = GenPropellerActuator(
                num_envs=num_envs,
                device=device,
                dt=dt,
                cfg=cfg.thruster_cfg,
            )
            self.thruster_cmd_dim = self.thruster_actuator.total_cmd_dim
        else:
            self.thruster_actuator = None
            self.thruster_cmd_dim = 0

        # ------------------------------------------------------------
        # Instantiate foil subsystem
        # ------------------------------------------------------------
        if cfg.foil_cfg is not None:
            self.foil_actuator = GenFoilActuator(
                num_envs=num_envs,
                device=device,
                dt=dt,
                cfg=cfg.foil_cfg,
                dynamics_list=cfg.foil_dynamics,
            )
            self.foil_cmd_dim = self.foil_actuator.total_cmd_dim
        else:
            self.foil_actuator = None
            self.foil_cmd_dim = 0

        # ------------------------------------------------------------
        # Total action dimension
        # ------------------------------------------------------------
        self.total_cmd_dim = self.thruster_cmd_dim + self.foil_cmd_dim

        # Combined wrench
        self._combined_forces = torch.zeros(
            (num_envs, 6), dtype=torch.float32, device=device
        )

    # ------------------------------------------------------------
    # Set target commands
    # ------------------------------------------------------------
    def set_target_cmd(self, actions):
        if actions.shape[1] != self.total_cmd_dim:
            raise ValueError(f"Expected {self.total_cmd_dim} actions, got {actions.shape[1]}")

        # Thrusters
        if self.thruster_actuator:
            thr_cmds = actions[:, :self.thruster_cmd_dim]
            self.thruster_actuator.set_target_cmd(thr_cmds)

        # Foils
        if self.foil_actuator:
            foil_cmds = actions[:, self.thruster_cmd_dim:]
            self.foil_actuator.set_target_cmd(foil_cmds)

    # ------------------------------------------------------------
    # Update all actuators
    # ------------------------------------------------------------
    def update(self, robot_heading_w, robot_lin_vel_b, foil_joint_positions=None):
        # Thrusters
        if self.thruster_actuator:
            self.thruster_forces = self.thruster_actuator.update_forces()
        else:
            self.thruster_forces = None

        # Foils
        if self.foil_actuator:
            if foil_joint_positions is None:
                raise ValueError("Foil joint positions required for foil update")
            self.foil_actuator.update_joint_cmds(foil_joint_positions)
            self.foil_forces = self.foil_actuator.update_forces(
                robot_heading_w, robot_lin_vel_b
            )
        else:
            self.foil_forces = None

    # ------------------------------------------------------------
    # Get combined wrench
    # ------------------------------------------------------------
    def get_forces(self):
        F = torch.zeros_like(self._combined_forces)

        if self.thruster_forces is not None:
            F += self.thruster_forces.sum(dim=1)

        if self.foil_forces is not None:
            F += self.foil_forces.sum(dim=1)

        self._combined_forces = F
        return F

    # ------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------
    def reset(self, env_ids=None):
        if self.thruster_actuator:
            self.thruster_actuator.reset(env_ids)
        if self.foil_actuator:
            self.foil_actuator.reset(env_ids)

        if env_ids is None:
            self._combined_forces.zero_()
        else:
            self._combined_forces[env_ids] = 0.0
