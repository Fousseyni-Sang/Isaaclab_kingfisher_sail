# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.physics.foil_dynamics import FoilDynamics

@configclass
class FoilActuatorCfg:

    cmd_lower_range: float = MISSING
    cmd_upper_range: float = MISSING
    command_rate: float = MISSING  # Frequency of command updates in Hz / max speed of foil movement in rad/s
    resolution: float = MISSING  # min discrete step in degrees
    precision: float = MISSING  # Precision for force calculations
    scale_joint_pos: float = MISSING  # Scale factor to convert command to joint position (rads)
    pos_from_com:tuple = MISSING  # position [dx, dy, dz] relative to center of mass (m)
    foil_type: str = "generic" # Optional: label/type (sail, keel, rudder, etc.)


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
        self.resolution = cfg.resolution*torch.pi/180 # normalized resolution (e.g. 1.8°/step)
        self.step_accuracy = self.precision * (torch.pi*self.resolution/180)
        self.scale_joint_pos = cfg.scale_joint_pos #(rads) # - small value to avoid exactly pi radians
        self.pos_from_com = torch.tensor(cfg.pos_from_com, device=self.device).repeat(self.num_envs,1)  # distance [dx, dy, dz] from foil to center of mass (m)
        # Constants
        self._max_cmd_delta = cfg.command_rate * dt

        # Initialize tensors
        self._current_cmd = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self._target_cmd = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)

        self.aero_forces = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.joint_position = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

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
                self.pos_from_com = torch.tensor(position, device=self.device).repeat(self.num_envs,1) + noise
            else:
                self.pos_from_com = torch.tensor(position, device=self.device).repeat(self.num_envs,1)  
    
    def get_forces_and_torques(self):
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
        #foil_angle = self._current_cmd * self.scale_joint_pos
        foil_angle = torch.atan2(torch.sin(self._current_cmd), torch.cos(self._current_cmd))

        return foil_angle
    
    def update_joint_cmd(self, current_joint_pos:torch.Tensor, target_cmd:torch.Tensor):
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
        if self.cfg.foil_type=="keel":
            return
        # update the current command based on the target command and maximum delta and the resolution
        scaled_command = self.scale_joint_pos*target_cmd
        self._current_cmd = torch.atan2(torch.sin(current_joint_pos.clone()), torch.cos(current_joint_pos.clone())) 

        self._target_cmd = torch.round(scaled_command / self.resolution)*self.resolution
        
        delta = torch.clamp(self._target_cmd - self._current_cmd, -self._max_cmd_delta, self._max_cmd_delta)
        step_error = (torch.rand_like(delta) * 2 - 1)*self.step_accuracy # random noise in range [-precision, precision]
        step_error = step_error*(delta!=0)  # only add error if there is a movement
        #print(f"self._current_cmd: {self._current_cmd} \ntarget_cmd: {self._target_cmd} \ndelta: {delta} \nstep_error:{step_error}")
        self._current_cmd += (delta + step_error)
        
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
        rd = (180/torch.pi)
        #print(f"aoa: {rd*angle_of_attack} foil_angle: {rd*self.dynamics.foil_angle} flow_angle: {rd*self.dynamics.apparent_flow_angle}")
        self.aero_forces = self.dynamics.compute_flow_effect(
            Uw=self.dynamics.Uw, Beta_w=self.dynamics.Beta_w, ship_heading_w=robot_heading_w,
            ship_lin_vel2D=robot_lin_vel_b[:, :2], foil_angle=self.dynamics.foil_angle, 
            angle_of_attack=self.dynamics.angle_of_attack,
        )
        
        return
        
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._current_cmd[env_ids, :] = 0.0
        self._target_cmd[env_ids, :] = 0.0
        self.aero_forces[env_ids, :] = 0.0



