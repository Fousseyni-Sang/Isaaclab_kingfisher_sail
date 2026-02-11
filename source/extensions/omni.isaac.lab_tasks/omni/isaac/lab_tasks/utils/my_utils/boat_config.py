from omni.isaac.lab.physics.foil_dynamics import FoilDynamicsCfg
from omni.isaac.lab.actuator_force.foil_actuator_force import FoilActuatorCfg
from omni.isaac.lab.physics.hydrostatics import HydrostaticsCfg
from omni.isaac.lab.physics.hydrodynamics import HydrodynamicsCfg
from omni.isaac.lab.actuator_force.actuator_force import PropellerActuatorCfg
from omni.isaac.lab.actuator_force.generic_actuator_force import ThrusterCfg, GenPropellerActuatorCfg
from omni.isaac.lab.actuator_force.generic_foil_actuator_force import GenFoilActuatorCfg
from omni.isaac.lab.actuator_force.robot_actuator_system import RobotActuatorSystemCfg
import torch
import random
from .constant import *
import os
import yaml

#------------------------------------------------------------
# FOIL DYNAMIC CONFIG
#------------------------------------------------------------
def rudder_config():
    
    # Rudder Hydrodynamics
    cfg: FoilDynamicsCfg = FoilDynamicsCfg()
    cfg.flow_density = 997.0
    cfg.foil_span = 0.2
    cfg.foil_chord = 0.021
    cfg.flow_direction = 30*torch.pi/180
    cfg.flow_speed = 0.05
    cfg.angle_of_attack = 20*torch.pi/180
    cfg.min_upflow_angle = 45*torch.pi/180
    cfg.max_downflow_angle = 140*torch.pi/180
    cfg.Reynold = 1000000  # based on flow_speed, chord, density and viscosity
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


def hydrodynamics_config(coeff_lateral_drag=1.0):
    c = coeff_lateral_drag
    # Hydrdynamics
    cfg: HydrodynamicsCfg = HydrodynamicsCfg()
    # linear Nominal [16.44998712, 15.79776044, 100, 13, 13, 6]
    # linear SID [0.0, 99.99, 99.99, 13.0, 13.0, 0.82985084]
    cfg.linear_damping = [0.0, 99.99*c, 99.99, 13.0, 13.0, 5.83] # not 5
    # quadratic Nominal [2.942, 2.7617212, 10, 5, 5, 5]
    # quadratic SID [17.257603, 99.99, 10.0, 5.0, 5.0, 17.33600724]
    cfg.quadratic_damping = [17.257603, 99.99*c, 10.0, 5.0, 5.0, 17.33600724]
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

def single_motor_cfg(holon=False):

    center_thr_cfg = ThrusterCfg(
            forces=KINGFISHER_THRUSTER_EXPERIMENT_VALUES if not holon else JELLYFISH_THRUSTER_EXPERIMENT_VALUES,
            interp_resolution=1000,
            pos_from_com=(0.0, 0., -0.16),
            holonomic=holon,
            positive_only=False,
            direction=(1.0, 0.0, 0.0),
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
            forces=JELLYFISH_THRUSTER_EXPERIMENT_VALUES,  # example curve
            interp_resolution=1000,
            pos_from_com=(-0.53, +0.37765, -0.16),
            holonomic=True,
            positive_only=True,
            direction=(1.0, 0.0, 0.0),
        )
    
    # 2 unidirectional forward thruster, forward and backward
    rear_right_thr_cfg = ThrusterCfg(
            forces=JELLYFISH_THRUSTER_EXPERIMENT_VALUES,
            interp_resolution=1000,
            pos_from_com=(-0.53, -0.37765, -0.16),
            holonomic=True,
            positive_only=True,
            direction=(1.0, 0.0, 0.0),
        )
    
    # 3 unidirectionnal (non-holonomic) thruster, forward and backward
    front_left_thr_cfg = ThrusterCfg(
            forces=JELLYFISH_THRUSTER_EXPERIMENT_VALUES,  # example curve
            interp_resolution=1000,
            pos_from_com=(0.53, +0.37765, -0.16),
            holonomic=True,
            positive_only=True,
            direction=(1.0, 0.0, 0.0),
        )
    
    # 4 unidirectional forward thruster, forward and backward
    front_right_thr_cfg = ThrusterCfg(
            forces=JELLYFISH_THRUSTER_EXPERIMENT_VALUES,
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
        foils=[rudder_cfg, keel_cfg,] 
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


def make_base_spec(base):

    if base=="jellyfish":
        return SPEC_JELLIFISH
    elif base=="kingfisher":
        return SPEC_KINGFISHER
    elif base=="single_motor":
        return SPEC_SINGLE_MOTOR
    elif base=="vap2":
        return SPEC_VAP2
    elif base=="vap4":
        return SPEC_VAP4
    elif base=="single_motor_sailboat":
        return SPEC_SAILBOAT
    else:
        raise ValueError("Available types are: {jellyfish, kingfisher, single_motor, " \
        "vap2, vap4, single_motor_sailboat}, got: "+f"{base}")

# --- 1) Parametric thruster curves ---------------------------------

def make_thruster_curve(base_forces, scale=1.0, bias=0.0, clip=None):
    forces = [scale * f + bias for f in base_forces]
    if clip is not None:
        lo, hi = clip
        forces = [max(lo, min(hi, f)) for f in forces]
    return forces

def get_layout(thruster_layout:str):
    if thruster_layout=="symmetric":
        return JELLYFISH_THRUSTER_EXPERIMENT_VALUES
    elif thruster_layout=="asymmetric":
        return KINGFISHER_THRUSTER_EXPERIMENT_VALUES
    else:
        raise ValueError(f"{thruster_layout} not in {'asymmetric', 'symmetric'}")
    

def load_spec():
    spec_path = os.environ.get("LL_OUTPUT_DIR")
    if spec_path is None:
        raise RuntimeError("LL_OUTPUT_DIR not set")
    spec_file = os.path.join(spec_path, "spec.yaml")
    with open(spec_file, "r") as f:
        return yaml.safe_load(f)


def make_boat_model(model_spec):
    # 1. Thrusters
    thrusters = []
    for i in range(model_spec["num_thrusters"]):
        thrusters.append(
            ThrusterCfg(
                forces=make_thruster_curve(
                    base_forces=get_layout(model_spec["thruster_layout"][i]),

                    scale=model_spec["thruster_curve_scale"],
                    bias=model_spec["thruster_curve_bias"]
                ),
                interp_resolution=1000,
                pos_from_com=model_spec["thruster_positions"][i],
                holonomic=model_spec["holonomic"][i],
                positive_only=model_spec["positive_only"][i],
                direction=model_spec["thruster_directions"][i],
            )
        )

    thruster_cfg = GenPropellerActuatorCfg(
        cmd_lower_range=-1.0,
        cmd_upper_range=1.0,
        command_rate=1.0,
        thrusters=thrusters
    )

    # 2. Foils
    foils = []
    if model_spec["has_rudder"]:
        foils.append(rudder_actuator_config())
    if model_spec["has_keel"]:
        foils.append(keel_actuator_config())
    if model_spec["has_sail"]:
        foils.append(sail_actuator_config())

    foil_cfg = GenFoilActuatorCfg(foils=foils) if foils else None

    # 3. Hydrodynamics
    hydro_cfg = hydrodynamics_config(coeff_lateral_drag=model_spec["hydro_lateral_drag"])
    
    # 4. Final robot system
    return RobotActuatorSystemCfg(
        thruster_cfg=thruster_cfg,
        foil_cfg=foil_cfg,
        hydrodynamics_cfg=hydro_cfg,
        model_spec=model_spec,
    )

def generate_boat_specs(num_specs=50):
    specs = []
    for _ in range(num_specs):
        base = random.choice(BASE_TYPES)

        spec = make_base_spec(base)

        # Add variations
        spec["thruster_curve_scale"] = random.uniform(0.5, 2.0)
        spec["thruster_curve_bias"] = random.uniform(-2.0, 2.0)
        spec["hydro_lateral_drag"] = random.uniform(0.05, 0.5)

        # Randomly toggle holonomic / positive_only
        for j in range(spec["num_thrusters"]):
            spec["holonomic"][j] = random.choice([True, False])
            spec["positive_only"][j] = random.choice([True, False])

        specs.append(spec)

    return specs
    
def jellyfish_roboat_cfg():

    """cfg = RobotActuatorSystemCfg(
        thruster_cfg=jellyfish_propeller_config(), 
        foil_cfg=None,
        hydrodynamics_cfg=hydrodynamics_config(LATERAL_DRAG_COEF),
    )"""

    return make_boat_model(make_base_spec("jellyfish"))

def kingfisher_roboat_cfg():
    cfg = RobotActuatorSystemCfg(
        thruster_cfg=kingfisher_propeller_config(), 
        foil_cfg=None
    )

    return cfg

def vap2_roboat_cfg():
    cfg = RobotActuatorSystemCfg(
        thruster_cfg=vap2_propeller_config(), 
        foil_cfg=None,
        hydrodynamics_cfg=hydrodynamics_config(LATERAL_DRAG_COEF)
    )

    return cfg

def vap4_roboat_cfg():
    cfg = RobotActuatorSystemCfg(
        thruster_cfg=vap4_propeller_config(), 
        foil_cfg=None,
        hydrodynamics_cfg=hydrodynamics_config(LATERAL_DRAG_COEF)
    )

    return cfg

def single_motor_roboat_cfg():

    """cfg = RobotActuatorSystemCfg(
        thruster_cfg=single_motor_cfg(), 
        foil_cfg=rudder_keel_cfg()
    )"""

    return make_boat_model(make_base_spec("single_motor"))

def single_motor_sailboat_cfg():

    cfg = RobotActuatorSystemCfg(
        thruster_cfg=single_motor_cfg(), 
        foil_cfg=rudder_keel_sail_cfg()
    )

    return cfg

