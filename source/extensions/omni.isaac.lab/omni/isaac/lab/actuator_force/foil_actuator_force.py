# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.physics.foil_model import FoilDynamics

@configclass
class FoilActuatorCfg:

    cmd_lower_range: float = MISSING
    cmd_upper_range: float = MISSING
    command_rate: float = MISSING  # Frequency of command updates in Hz
    resolution: float = MISSING  # min discrete step in degrees
    precision: float = MISSING  # Precision for force calculations
    scale_joint_pos: float = MISSING  # Scale factor to convert command to joint position (rads)
    pos_from_com:tuple = MISSING  # distance [dx, dy, dz] from foil to center of mass (m)


class FoilActuator:
    def __init__(self, num_envs, dynamics:FoilDynamics, dt:float, cfg: FoilActuatorCfg = FoilActuatorCfg()):
        """
        Initializes the StepperActuator class with the given parameters.

        Args:
            num_envs (int): Number of environments.
            device (torch.device): The device (CPU or GPU) on which tensors will be allocated.
            dt (float): Time step duration.
            cfg (StepperActuatorCfg, optional): Configuration object for the propeller actuator. Defaults to StepperActuatorCfg().

        Attributes:
            num_envs (int): Number of environments.
            device (torch.device): The device (CPU or GPU) on which tensors are allocated.
            dt (float): Time step duration.
            cfg (StepperActuatorCfg): Configuration object for the propeller actuator.
            _max_cmd_delta (float): Maximum command delta calculated from command rate and time step.
            _current_cmd (torch.Tensor): Tensor to store current commands for each environment.
            _target_cmd (torch.Tensor): Tensor to store target commands for each environment.
            foil_forces (torch.Tensor): Tensor to store foil aero/hydro forces for each environment.
            
        """
        
        self.cfg = cfg
        self.dynamics = dynamics
        self.num_envs = num_envs
        self.device = dynamics.device
        self.dt = dt
        self.precision = cfg.precision
        self.resolution = cfg.resolution/180 # normalized resolution (e.g. 1.8°/step -> 0.01)
        self.scale_joint_pos = cfg.scale_joint_pos - self.resolution #(rads) # - small value to avoid exactly pi radians
        self.pos_from_com = torch.tensor(cfg.pos_from_com, device=self.device).repeat(self.num_envs,1)  # distance [dx, dy, dz] from foil to center of mass (m)
        # Constants
        self._max_cmd_delta = cfg.command_rate * dt

        # Initialize tensors
        self._current_cmd = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self._target_cmd = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)

        self.aero_forces = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.joint_psoition = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.reset()

    
    @property      
    def current_cmd(self):
        """
        Returns the current command for each environment.

        Returns:
            torch.Tensor: A tensor containing the current commands.
        """
        return self._current_cmd
    
    @property
    def target_cmd(self):
        """
        Returns the target command for each environment.

        Returns:
            torch.Tensor: A tensor containing the target commands.
        """
        return self._target_cmd
    
    @property
    def max_cmd_delta(self):
        """
        Returns the maximum command delta.

        Returns:
            float: The maximum command delta.
        """
        return self._max_cmd_delta
    
    def set_pos_from_com(self, position:list|None=None, randomize:bool=False):
        """ Set the position of the foil from the center of mass of the boat 
        Args:
            position (list, optional): list of 3 elements [dx, dy, dz]. Defaults to None.
        """
        if position is not None:
            if randomize:
                noise = (torch.rand((self.num_envs,3), device=self.device)-0.5)*0.1  # random noise in range [-0.05, 0.05]
                self.pos_from_com = (torch.tensor(position, device=self.device).repeat(self.num_envs,1) + noise)
            else:
                self.pos_from_com = torch.tensor(position, device=self.device).repeat(self.num_envs,1)  
    
    def get_forces(self):
        """
        Computes the forces and torques generated on the boat by the foil in boat body frame.

        Returns:
            torch.Tensor: A tensor containing the  forces and torques.
        """

        forces_torques = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        forces_torques[:, 0:3] = self.aero_forces  # Forces in X, Y, Z
        
        torque = torch.cross(self.pos_from_com, self.aero_forces, dim=1)
        forces_torques[:, 3:6] = torque  # Torques in Roll, Pitch, Yaw

        return forces_torques
        
    
    def get_joint_positions(self):
        """
        Computes the foil angles based on the current commands.
        Updates the dynamics foil angle attribute.
        Returns:
            torch.Tensor: A tensor containing the foil angles for each environment.
        """
        foil_angle = self._current_cmd * self.scale_joint_pos
        foil_angle = torch.atan2(torch.sin(foil_angle), torch.cos(foil_angle))

        return foil_angle
    
    def update_joint_cmd(self, command):
        """
        Updates the current commands based on the target commands and maximum delta, and calculates the thruster forces.

        This method performs the following steps:
        1. Quantizes the target commands to the nearest valid step based on the resolution.
        2. Computes the difference between the target commands and the current commands.
        3. Clamps the difference to be within the range defined by the maximum command delta.
        4. Updates the current commands by adding the clamped difference.
        
        Returns:
            torch.Tensor: The updated thruster forces.
        """
        # update the current command based on the target command and maximum delta and the resolution
         
        self._target_cmd = torch.round(command / self.resolution) * self.resolution
        delta = torch.clamp(self._target_cmd - self._current_cmd, -self._max_cmd_delta, self._max_cmd_delta)
        random_noise = torch.randn_like(self._current_cmd)*self.precision
        self._current_cmd += (delta + random_noise*delta)

        return

    def update_forces(self, robot_heading_w, robot_lin_vel_b):

        """ Updates the foil forces based on the current joint positions and dynamics.   
        Args:
            robot: The robot object containing the necessary data for force computation.
        Returns:
            torch.Tensor: The updated foil forces.
        """
        
        self.dynamics.foil_angle = self.get_joint_positions()
        angle_of_attack = self.dynamics.get_angle_of_attack(self.dynamics.apparent_flow_angle,
                                        self.dynamics.foil_angle)
        self.dynamics.angle_of_attack = angle_of_attack.clone()
        
        self.aero_forces = self.dynamics.compute_flow_effect(
            Uw=self.dynamics.Uw, Beta_w=self.dynamics.Beta_w, ship_heading_w=robot_heading_w,
            ship_lin_vel2D=robot_lin_vel_b[:, :2], foil_angle=self.dynamics.foil_angle, 
            angle_of_attack=self.dynamics.angle_of_attack,
        )
        
        return
        
    def reset(self):
        self._current_cmd[:, :] = 0.0
        self._target_cmd[:, :] = 0.0



