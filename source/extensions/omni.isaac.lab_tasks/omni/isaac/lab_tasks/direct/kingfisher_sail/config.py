from omni.isaac.lab.physics.foil_model import FoilDynamicsCfg
from omni.isaac.lab.actuator_force.foil_actuator_force import FoilActuatorCfg
from omni.isaac.lab.physics.hydrostatics import HydrostaticsCfg
from omni.isaac.lab.physics.hydrodynamics import HydrodynamicsCfg
from omni.isaac.lab.actuator_force.actuator_force import PropellerActuatorCfg
import torch

def rudder_config():
    
    # Rudder Hydrodynamics
    cfg: FoilDynamicsCfg = FoilDynamicsCfg()
    cfg.flow_density = 997.0
    cfg.foil_span = 1
    cfg.foil_chord = 0.2
    cfg.flow_direction = -179*torch.pi/180
    cfg.flow_speed = 5
    cfg.angle_of_attack = 20*torch.pi/180
    cfg.min_upflow_angle = 45*torch.pi/180
    cfg.max_downflow_angle = 140*torch.pi/180
    cfg.Reynold = 500000  # based on flow_speed, chord, density and viscosity
    return cfg

def sail_config():

    # Aerodynamics
    cfg: FoilDynamicsCfg = FoilDynamicsCfg()
    cfg.flow_density = 1.225
    cfg.foil_span = 1
    cfg.foil_chord = 0.2
    cfg.flow_direction = -179*torch.pi/180
    cfg.flow_speed = 5
    cfg.angle_of_attack = 20*torch.pi/180
    cfg.min_upflow_angle = 45*torch.pi/180
    cfg.max_downflow_angle = 140*torch.pi/180
    cfg.Reynold = 200000  # based on flow_speed, chord, density and viscosity

    return cfg

def keel_config():

    # keel Hydrodynamics
    cfg: FoilDynamicsCfg = FoilDynamicsCfg()
    cfg.flow_density = 997.0
    cfg.foil_span = 1
    cfg.foil_chord = 0.2
    cfg.flow_direction = -179*torch.pi/180
    cfg.flow_speed = 5
    cfg.angle_of_attack = 20*torch.pi/180
    cfg.min_upflow_angle = 0.0
    cfg.max_downflow_angle = 0.0
    cfg.Reynold = 200000  # based on flow_speed, chord, density and viscosity

    return cfg

def hydrostatics_config():
    
    # Hydrostatics
    cfg: HydrostaticsCfg = HydrostaticsCfg()
    cfg.mass = 35.0  # Kg considering added sensors
    cfg.width = 1.0  # Kingfisher/Heron width 1.0m in Spec Sheet
    cfg.length = 1.3  # Kingfisher/Heron length 1.3m in Spec Sheet
    cfg.waterplane_area = 0.33  # 0.15 width * 1.1 length * 2 hulls
    cfg.draught_offset = 0.21986  # Distance from base_link to bottom of the hull
    cfg.max_draught = 0.20  # Kingfisher/Heron draught 120mm in Spec Sheet
    cfg.average_hydrostatics_force = 275.0
    
    return cfg

def hydrodynamics_config():
    
    # Hydrdynamics
    cfg: HydrodynamicsCfg = HydrodynamicsCfg()
    # linear Nominal [16.44998712, 15.79776044, 100, 13, 13, 6]
    # linear SID [0.0, 99.99, 99.99, 13.0, 13.0, 0.82985084]
    cfg.linear_damping = [0.0, 99.99, 99.99, 13.0, 13.0, 5.83] # not 5
    # quadratic Nominal [2.942, 2.7617212, 10, 5, 5, 5]
    # quadratic SID [17.257603, 99.99, 10.0, 5.0, 5.0, 17.33600724]
    cfg.quadratic_damping = [17.257603, 99.99, 10.0, 5.0, 5.0, 17.33600724]
    cfg.use_drag_randomization = False
    cfg.linear_damping_rand = [0.1, 0.1, 0.0, 0.0, 0.0, 0.1]
    cfg.quadratic_damping_rand = [0.1, 0.1, 0.0, 0.0, 0.0, 0.1]
    return cfg

def propeller_config():
    
    cfg: PropellerActuatorCfg = PropellerActuatorCfg()
    cfg.cmd_lower_range = -1.0
    cfg.cmd_upper_range = 1.0
    cfg.command_rate = (cfg.cmd_upper_range - cfg.cmd_lower_range) / 2.0
    cfg.forces_left = [
        -4.0,  # -1.0
        -4.0,  # -0.9
        -4.0,  # -0.8
        -4.0,  # -0.7
        -2.0,  # -0.6
        -1.0,  # -0.5
        0.0,  # -0.4
        0.0,  # -0.3
        0.0,  # -0.2
        0.0,  # -0.1
        0.0,  # 0.0
        0.0,  # 0.1
        0.0,  # 0.2
        0.5,  # 0.3
        1.5,  # 0.4
        4.75,  # 0.5
        8.25,  # 0.6
        16.0,  # 0.7
        19.5,  # 0.8
        19.5,  # 0.9
        19.5,  # 1.0
    ]
    cfg.forces_right = cfg.forces_left
    return cfg

def rudder_actuator_config():
    
    cfg: FoilActuatorCfg = FoilActuatorCfg()
    cfg.cmd_lower_range = -1.0
    cfg.cmd_upper_range = 1.0
    cfg.command_rate = (cfg.cmd_upper_range - cfg.cmd_lower_range) / 2.0
    cfg.resolution = 1.8  # degrees
    cfg.precision = 0.01  # radians
    cfg.scale_joint_pos = torch.pi/6  # radians per command unit
    cfg.pos_from_com = (0.0, 0.0, 0.0)  # meters
    return cfg

def sail_actuator_config():
    
    cfg: FoilActuatorCfg = FoilActuatorCfg()
    cfg.cmd_lower_range = -1.0
    cfg.cmd_upper_range = 1.0
    cfg.command_rate = (cfg.cmd_upper_range - cfg.cmd_lower_range) / 2.0
    cfg.resolution = 1.8  # degrees
    cfg.precision = 0.01  # radians
    cfg.scale_joint_pos = torch.pi/6  # radians per command unit
    cfg.pos_from_com = (0.0, 0.0, 0.0)  # meters
    return cfg

def keel_actuator_config():
    
    cfg: FoilActuatorCfg = FoilActuatorCfg()
    cfg.cmd_lower_range = -1.0
    cfg.cmd_upper_range = 1.0
    cfg.command_rate = (cfg.cmd_upper_range - cfg.cmd_lower_range) / 2.0
    cfg.resolution = 1.8  # degrees
    cfg.precision = 0.01  # radians
    cfg.scale_joint_pos = torch.pi/6  # radians per command unit
    cfg.pos_from_com = (0.0, 0.0, 0.0)  # meters
    return cfg