from omni.isaac.lab.physics.foil_dynamics import FoilDynamicsCfg
from omni.isaac.lab.actuator_force.foil_actuator_force import FoilActuatorCfg
from omni.isaac.lab.physics.hydrostatics import HydrostaticsCfg
from omni.isaac.lab.physics.hydrodynamics import HydrodynamicsCfg
from omni.isaac.lab.actuator_force.actuator_force import PropellerActuatorCfg
from omni.isaac.lab.actuator_force.generic_actuator_force import ThrusterCfg, GenPropellerActuatorCfg
from omni.isaac.lab.actuator_force.generic_foil_actuator_force import GenFoilActuatorCfg
from omni.isaac.lab.actuator_force.robot_actuator_system import RobotActuatorSystemCfg
import torch

KINGFISHER_THRUSTER_EXPERIMENT_VALUES = [
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
# Jellyfish negative thrust is the same magnitude as its positive thrust
JELLYFISH_THRUSTER_EXPERIMENT_VALUES = [
        -19.5, 
        -19.5,
        -19.5,  # -1.0
        -16.0,  # -0.9
        -8.25,  # -0.8
        -4.75,  # -0.7
        -1.5,  # -0.6
        -0.5,  # -0.5
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

#------------------------------------------------------------
# FOIL DYNAMIC CONFIG
#------------------------------------------------------------
def rudder_config():
    
    # Rudder Hydrodynamics
    cfg: FoilDynamicsCfg = FoilDynamicsCfg()
    cfg.flow_density = 997.0
    cfg.foil_span = 1
    cfg.foil_chord = 0.2
    cfg.flow_direction = -179*torch.pi/180
    cfg.flow_speed = 0.05
    cfg.angle_of_attack = 20*torch.pi/180
    cfg.min_upflow_angle = 45*torch.pi/180
    cfg.max_downflow_angle = 140*torch.pi/180
    cfg.Reynold = 500000  # based on flow_speed, chord, density and viscosity
    return cfg

def keel_config():

    # keel Hydrodynamics
    cfg: FoilDynamicsCfg = FoilDynamicsCfg()
    cfg.flow_density = 997.0
    cfg.foil_span = 1
    cfg.foil_chord = 0.2
    cfg.flow_direction = -179*torch.pi/180
    cfg.flow_speed = 0.05
    cfg.angle_of_attack = 20*torch.pi/180
    cfg.min_upflow_angle = 0.0
    cfg.max_downflow_angle = 0.0
    cfg.Reynold = 200000  # based on flow_speed, chord, density and viscosity

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

#------------------------------------------------------------
# HYDRODYNAMIC & HYDROSTATIC CONFIG
#------------------------------------------------------------
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

#------------------------------------------------------------
# PROPELLER/THRUSTER DYNAMIC CONFIG
#------------------------------------------------------------
def propeller_config():
    
    cfg: PropellerActuatorCfg = PropellerActuatorCfg()
    cfg.cmd_lower_range = -1.0
    cfg.cmd_upper_range = 1.0
    cfg.command_rate = (cfg.cmd_upper_range - cfg.cmd_lower_range) / 2.0
    cfg.forces_left = KINGFISHER_THRUSTER_EXPERIMENT_VALUES
    cfg.forces_right = cfg.forces_left
    return cfg

def kingfisher_propeller_config():

    # 1 unidirectionnal (non-holonomic) thruster, forward and backward
    left_thr_cfg = ThrusterCfg(
            forces=KINGFISHER_THRUSTER_EXPERIMENT_VALUES,  # example curve
            interp_resolution=1000,
            pos_from_com=(-0.53, 0.37765, -0.16),
            holonomic=False,
            positive_only=False,
            direction=(1.0, 0.0, 0.0),
        )
    
    # 2 unidirectional forward thruster, forward and backward
    right_thr_cfg = ThrusterCfg(
            forces=KINGFISHER_THRUSTER_EXPERIMENT_VALUES,
            interp_resolution=1000,
            pos_from_com=(-0.53, -0.37765, -0.16),
            holonomic=False,
            positive_only=False,
            direction=(1.0, 0.0, 0.0),
        )
    
    thrust_cfg = GenPropellerActuatorCfg(
    cmd_lower_range=-1.0,
    cmd_upper_range=1.0,
    command_rate=1.0,

    thrusters=[left_thr_cfg, right_thr_cfg]    
    )

    return thrust_cfg

def jellyfish_propeller_config():

    # 1 unidirectionnal (non-holonomic) thruster, forward and backward
    left_thr_cfg = ThrusterCfg(
            forces=KINGFISHER_THRUSTER_EXPERIMENT_VALUES,  # example curve
            interp_resolution=1000,
            pos_from_com=(0.0, +0.37765, -0.16),
            holonomic=False,
            positive_only=False,
            direction=(1.0, 0.0, 0.0),
        )
    
    # 2 unidirectional forward thruster, forward and backward
    right_thr_cfg = ThrusterCfg(
            forces=KINGFISHER_THRUSTER_EXPERIMENT_VALUES,
            interp_resolution=1000,
            pos_from_com=(0.0, -0.37765, -0.16),
            holonomic=False,
            positive_only=False,
            direction=(1.0, 0.0, 0.0),
        )
    
    # 3 lateral unidirectional, left and right (jellyfish style)
    center_thr_cfg = ThrusterCfg(
            forces=JELLYFISH_THRUSTER_EXPERIMENT_VALUES,
            interp_resolution=1000,
            pos_from_com=(0.0, 0., -0.16),
            holonomic=False,
            positive_only=False,
            direction=(0.0, 1.0, 0.0),
            )
    
    thrust_cfg = GenPropellerActuatorCfg(
    cmd_lower_range=-1.0,
    cmd_upper_range=1.0,
    command_rate=1.0,

    thrusters=[left_thr_cfg, right_thr_cfg, center_thr_cfg]    
    )

    return thrust_cfg

def vap2_propeller_config():

    # 1 unidirectionnal (non-holonomic) thruster, forward and backward
    left_thr_cfg = ThrusterCfg(
            forces=KINGFISHER_THRUSTER_EXPERIMENT_VALUES,  # example curve
            interp_resolution=1000,
            pos_from_com=(0.0, +0.37765, -0.16),
            holonomic=True,
            positive_only=True,
        )
    
    # 2 unidirectional forward thruster, forward and backward
    right_thr_cfg = ThrusterCfg(
            forces=KINGFISHER_THRUSTER_EXPERIMENT_VALUES,
            interp_resolution=1000,
            pos_from_com=(0.0, -0.37765, -0.16),
            holonomic=True,
            positive_only=True,
        )
    
    thrust_cfg = GenPropellerActuatorCfg(
    cmd_lower_range=-1.0,
    cmd_upper_range=1.0,
    command_rate=1.0,

    thrusters=[left_thr_cfg, right_thr_cfg]    
    )

    return thrust_cfg

def single_motor_cfg():

    center_thr_cfg = ThrusterCfg(
            forces=JELLYFISH_THRUSTER_EXPERIMENT_VALUES,
            interp_resolution=1000,
            pos_from_com=(0.0, 0., -0.16),
            holonomic=False,
            positive_only=False,
            direction=(0.0, 1.0, 0.0),
            )
    thrust_cfg = GenPropellerActuatorCfg(
    cmd_lower_range=-1.0,
    cmd_upper_range=1.0,
    command_rate=1.0,

    thrusters=[center_thr_cfg]    
    )

    return thrust_cfg


def vap4_propeller_config():

    # 1 unidirectionnal (non-holonomic) thruster, forward and backward
    rear_left_thr_cfg = ThrusterCfg(
            forces=KINGFISHER_THRUSTER_EXPERIMENT_VALUES,  # example curve
            interp_resolution=1000,
            pos_from_com=(-0.53, +0.37765, -0.16),
            holonomic=True,
            positive_only=True,
            direction=(1.0, 0.0, 0.0),
        )
    
    # 2 unidirectional forward thruster, forward and backward
    rear_right_thr_cfg = ThrusterCfg(
            forces=KINGFISHER_THRUSTER_EXPERIMENT_VALUES,
            interp_resolution=1000,
            pos_from_com=(-0.53, -0.37765, -0.16),
            holonomic=True,
            positive_only=True,
            direction=(1.0, 0.0, 0.0),
        )
    
    # 3 unidirectionnal (non-holonomic) thruster, forward and backward
    front_left_thr_cfg = ThrusterCfg(
            forces=KINGFISHER_THRUSTER_EXPERIMENT_VALUES,  # example curve
            interp_resolution=1000,
            pos_from_com=(0.53, +0.37765, -0.16),
            holonomic=True,
            positive_only=True,
            direction=(1.0, 0.0, 0.0),
        )
    
    # 4 unidirectional forward thruster, forward and backward
    front_right_thr_cfg = ThrusterCfg(
            forces=KINGFISHER_THRUSTER_EXPERIMENT_VALUES,
            interp_resolution=1000,
            pos_from_com=(0.53, -0.37765, -0.16),
            holonomic=True,
            positive_only=True,
            direction=(1.0, 0.0, 0.0),
        )
    
    thrust_cfg = GenPropellerActuatorCfg(
    cmd_lower_range=-1.0,
    cmd_upper_range=1.0,
    command_rate=1.0,

    thrusters=[rear_left_thr_cfg, rear_right_thr_cfg, front_left_thr_cfg, front_right_thr_cfg]    
    )

    return thrust_cfg

#------------------------------------------------------------
# FOIL ACUATOR CONFIG
#------------------------------------------------------------

def rudder_actuator_config():
    
    cfg: FoilActuatorCfg = FoilActuatorCfg()
    cfg.cmd_lower_range = -1.0
    cfg.cmd_upper_range = 1.0
    cfg.command_rate = (cfg.cmd_upper_range - cfg.cmd_lower_range) / 2.0
    cfg.resolution = 1.8  # degrees
    cfg.precision = 0.01  # radians
    cfg.scale_joint_pos = torch.pi  # radians per command unit
    cfg.pos_from_com = (-0.5, 0.0, -0.5)  # meters
    cfg.foil_type = "rudder"
    return cfg

def sail_actuator_config():
    
    cfg: FoilActuatorCfg = FoilActuatorCfg()
    cfg.cmd_lower_range = -1.0
    cfg.cmd_upper_range = 1.0
    cfg.command_rate = (cfg.cmd_upper_range - cfg.cmd_lower_range) / 2.0
    cfg.resolution = 1.8  # degrees
    cfg.precision = 0.01  # radians
    cfg.scale_joint_pos = torch.pi  # radians per command unit
    cfg.pos_from_com = (0.0, 0.0, 0.0)  # meters
    cfg.foil_type = "sail"
    return cfg

def keel_actuator_config():
    
    cfg: FoilActuatorCfg = FoilActuatorCfg()
    cfg.cmd_lower_range = -1.0
    cfg.cmd_upper_range = 1.0
    cfg.command_rate = (cfg.cmd_upper_range - cfg.cmd_lower_range) / 2.0
    cfg.resolution = 1.8  # degrees
    cfg.precision = 0.01  # radians
    cfg.scale_joint_pos = torch.pi  # radians per command unit
    cfg.pos_from_com = (0.0, 0.0, 0.0)  # meters
    cfg.foil_type = "keel"
    return cfg

#------------------------------------------------------------
# GENERIC FOIL ACTUATOR CONFIG
#------------------------------------------------------------

def rudder_keel_cfg():

    keel_cfg = keel_actuator_config()
    rudder_cfg = rudder_actuator_config()

    cfg = GenFoilActuatorCfg(
        foils=[rudder_cfg, keel_cfg]
    )

    return cfg

def rudder_keel_sail_cfg():

    keel_cfg = keel_actuator_config()
    rudder_cfg = rudder_actuator_config()
    sail_cfg = sail_actuator_config()

    cfg = GenFoilActuatorCfg(
        foils=[rudder_cfg, keel_cfg, sail_cfg]
    )

    return cfg

#------------------------------------------------------------
# ROBOT ACTUATOR SYSTEM CONFIG
#------------------------------------------------------------

def jellyfish_roboat():

    cfg = RobotActuatorSystemCfg(
        thruster_cfg=jellyfish_propeller_config(), 
        foil_cfg=None
    )

    return cfg

def kingfisher_roboat():
    cfg = RobotActuatorSystemCfg(
        thruster_cfg=kingfisher_propeller_config(), 
        foil_cfg=None
    )

    return cfg

def vap2_roboat():
    cfg = RobotActuatorSystemCfg(
        thruster_cfg=vap2_propeller_config(), 
        foil_cfg=None
    )

    return cfg

def vap4_roboat():
    cfg = RobotActuatorSystemCfg(
        thruster_cfg=vap4_propeller_config(), 
        foil_cfg=None
    )

    return cfg

def single_motor_boat():

    cfg = RobotActuatorSystemCfg(
        thruster_cfg=single_motor_cfg(), 
        foil_cfg=rudder_keel_cfg()
    )

    return cfg

def single_motor_sailboat():

    cfg = RobotActuatorSystemCfg(
        thruster_cfg=single_motor_cfg(), 
        foil_cfg=rudder_keel_sail_cfg()
    )

    return cfg