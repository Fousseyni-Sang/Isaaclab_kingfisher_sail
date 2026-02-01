from dataclasses import dataclass, field
from typing import List, Tuple
import torch

MISSING = None  # replace with your own

@dataclass
class ThrusterCfg:
    # Force curve as function of scalar command in [-1, 1] 
    forces: List[float] = field(default_factory=list)
    interp_resolution: int = 1000

    # Position of thruster relative to CoM (for torque computation)
    pos_from_com: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # If True: command is 2D [u, theta], omni-directional in plane
    # If False: command is 1D [u], fixed direction
    holonomic: bool = False

    # If False: command can be negative (bidirectional)
    # If True: command is assumed >= 0 (or clamped)
    positive_only: bool = False
    num_dim: int = 1

    # Fixed direction for non-holonomic thrusters (unit vector in 3D)
    # Example: (1, 0, 0) for forward-only, (0, 1, 0) for lateral, etc.
    direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    def __post_init__(self):
        self.num_dim = 2 if self.holonomic else 1


@dataclass
class GenPropellerActuatorCfg:
    cmd_lower_range: float = MISSING
    cmd_upper_range: float = MISSING
    command_rate: float = MISSING  # Hz

    # List of thrusters, arbitrary number and types
    thrusters: List[ThrusterCfg] = field(default_factory=list)
    num_thrusters: int = 0

    def __post_init__(self): 
        self.num_thrusters = len(self.thrusters)


class GenPropellerActuator:
    def __init__(self, num_envs, device, dt, cfg: GenPropellerActuatorCfg):
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.cfg = cfg
        self.num_thrusters = cfg.num_thrusters
        if self.num_thrusters == 0:
            raise ValueError("PropellerActuatorCfg.thrusters must contain at least one thruster.")

        # Command dynamics
        self._max_cmd_delta = cfg.command_rate * dt

        # Per-thruster command dimensionality (1D or 2D)
        self._thruster_cmd_dims = [
            thr_cfg.num_dim for thr_cfg in self.cfg.thrusters
        ]

        self.total_cmd_dim = sum(self._thruster_cmd_dims)
        
        # Flat command tensors: shape (num_envs, total_cmd_dim)
        self._current_cmds = torch.zeros(
            (self.num_envs, self.total_cmd_dim), dtype=torch.float32, device=self.device
        )
        self._target_cmds = torch.zeros_like(self._current_cmds)
        self.idx = None
        # Precompute interpolation tables per thruster
        self._interp_forces = []
        for thr_cfg in self.cfg.thrusters:
            forces = torch.tensor(thr_cfg.forces, dtype=torch.float32, device=self.device)
            interp = self.linear_interpolate_1d(forces, thr_cfg.interp_resolution)
            self._interp_forces.append(interp)

        # Thruster forces in world/body frame: (num_envs, num_thrusters, 3)
        self.thruster_forces = torch.zeros(
            (self.num_envs, self.num_thrusters, 3), dtype=torch.float32, device=self.device
        )

        # Optional: torques from thrusters about CoM: (num_envs, num_thrusters, 3)
        self.thruster_torques = torch.zeros_like(self.thruster_forces)

        # Precompute command index offsets for slicing
        self._cmd_offsets = []
        offset = 0
        for dim in self._thruster_cmd_dims:
            self._cmd_offsets.append(offset)
            offset += dim


        self.reset()

    def linear_interpolate_1d(self, x: torch.Tensor, size: int) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            x.view(1, 1, -1), size=size, mode="linear", align_corners=True
        ).squeeze()

    def _cmd_to_index(self, cmd: torch.Tensor, thr_cfg: ThrusterCfg) -> torch.Tensor:
        """
        Map scalar command in [cmd_lower_range, cmd_upper_range] to [0, interp_resolution-1].
        """
        cmd_clamped = torch.clamp(cmd, self.cfg.cmd_lower_range, self.cfg.cmd_upper_range)
        norm = (cmd_clamped - self.cfg.cmd_lower_range) / (
            self.cfg.cmd_upper_range - self.cfg.cmd_lower_range
        )
        idx = torch.round(norm * (thr_cfg.interp_resolution - 1)).to(torch.long)
        return idx

    def get_forces(self) -> torch.Tensor:
        """
        Compute forces for all thrusters and all envs.
        Returns:
            thruster_forces: (num_envs, num_thrusters, 3)
        """
        forces = torch.zeros_like(self.thruster_forces)
        
        for i, thr_cfg in enumerate(self.cfg.thrusters):
            offset = self._cmd_offsets[i]
            dim = self._thruster_cmd_dims[i]
            interp = self._interp_forces[i]
            
            if thr_cfg.holonomic:
                # Command: [u, theta]
                u = self._current_cmds[:, offset]       # magnitude command
                theta = self._current_cmds[:, offset+1] # direction angle (rad)

                if thr_cfg.positive_only:
                    u = torch.clamp(u, min=0.0)
                
                idx = self._cmd_to_index(u, thr_cfg)
                mag = interp[idx]  # (num_envs,)
                
                # 2D force in plane (x, y), z = 0
                fx = mag * torch.cos(theta)
                fy = mag * torch.sin(theta)
                fz = torch.zeros_like(fx)

                forces[:, i, 0] = fx
                forces[:, i, 1] = fy
                forces[:, i, 2] = fz

            else:
                # Command: [u], fixed direction
                u = self._current_cmds[:, offset]

                if thr_cfg.positive_only:
                    u = torch.clamp(u, min=0.0)
                
                idx = self._cmd_to_index(u, thr_cfg)
                
                mag = interp[idx]  # (num_envs,)
                
                direction = torch.tensor(
                    thr_cfg.direction, dtype=torch.float32, device=self.device
                )
                
                direction = direction / (direction.norm() + 1e-8)

                forces[:, i, :] = mag.unsqueeze(-1) * direction
        
        self.thruster_forces = forces
        #print(f"mag values: {self.thruster_forces}")
        self._update_torques()
        return self.thruster_forces

    def _update_torques(self):
        """
        Compute torque = r x F for each thruster.
        """
        torques = torch.zeros_like(self.thruster_torques)
        for i, thr_cfg in enumerate(self.cfg.thrusters):
            r = torch.tensor(
                thr_cfg.pos_from_com, dtype=torch.float32, device=self.device
            ).view(1, 3)  # (1, 3)
            F = self.thruster_forces[:, i, :]  # (num_envs, 3)

            # Cross product r x F
            rxF = torch.cross(r.expand_as(F), F, dim=-1)
            torques[:, i, :] = rxF
            
        self.thruster_torques = torques

    def update_forces(self) -> torch.Tensor:
        """
        Smoothly update current commands toward target commands and recompute forces.
        """
        delta = torch.clamp(
            self._target_cmds - self._current_cmds,
            -self._max_cmd_delta,
            self._max_cmd_delta,
        )
        self._current_cmds += delta
        return self.get_forces()

    def set_target_cmd(self, commands: torch.Tensor):
        """
        commands: (num_envs, total_cmd_dim)
        """
        if commands.shape != self._target_cmds.shape:
            raise ValueError(
                f"Expected commands of shape {self._target_cmds.shape}, got {commands.shape}"
            )
        self._target_cmds = commands.to(self.device)

    def reset(self):
        self._current_cmds.zero_()
        self._target_cmds.zero_()
        self.thruster_forces.zero_()
        self.thruster_torques.zero_()
