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

BASE_TYPES = [
    "kingfisher",
    "vap2",
    "jellyfish",
]
# "single_motor", "vap4",
#
#"single_motor_sailboat"
LATERAL_DRAG_COEF = 0.7

SPEC_JELLIFISH = {
        "type": "jellyfish",          # or "vap2", "vap4", "kingfisher", "single_motor", etc.
        "thruster_layout": ["asymmetric", "asymmetric", "symmetric"],
        "thruster_curve_scale": 1.,
        "thruster_curve_bias": 0.,
        "hydro_lateral_drag": 0.7,
        "num_thrusters": 3,
        "holonomic": [False, False, False],
        "thruster_positions": [[0.0, +0.37765, -0.16], [0.0, -0.37765, -0.16], [0.0, 0.0, -0.16]],
        "thruster_directions": [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        "positive_only": [False, False, False],
        "has_rudder": False,
        "has_keel": False,
        "has_sail": False,
        }

SPEC_KINGFISHER = ({
        "type": "kingfisher",          # or "vap2", "vap4", "kingfisher", "single_motor", etc.
        "thruster_layout": ["asymmetric", "asymmetric"],
        "thruster_curve_scale": 1.,
        "thruster_curve_bias": 0.,
        "hydro_lateral_drag": 1.0,
        "num_thrusters": 2,
        "holonomic": [False, False],
        "thruster_positions": [[-0.53, 0.37765, -0.16], [-0.53, -0.37765, -0.16]],
        "thruster_directions": [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        "positive_only": [False, False],
        "has_rudder": False,
        "has_keel": False,
        "has_sail": False,
        })

SPEC_SINGLE_MOTOR = {
        "type": "single_motor",          # or "vap2", "vap4", "kingfisher", "single_motor", etc.
        "thruster_layout": ["asymmetric"],
        "thruster_curve_scale": 1.,
        "thruster_curve_bias": 0.,
        "hydro_lateral_drag": 1.,
        "num_thrusters": 1,
        "holonomic": [False],
        "thruster_positions": [[-0.53, 0.0, -0.16]],
        "thruster_directions": [(1.0, 0.0, 0.0)],
        "positive_only": [False],
        "has_rudder": True,
        "has_keel": True,
        "has_sail": False,
        }

SPEC_VAP2 = ({
        "type": "vap2",          # or "vap2", "vap4", "kingfisher", "single_motor", etc.
        "thruster_layout": ["symmetric", "symmetric"],
        "thruster_curve_scale": 1.,
        "thruster_curve_bias": 0.,
        "hydro_lateral_drag": 0.7,
        "num_thrusters": 2,
        "holonomic": [True, True],
        "thruster_positions": [[0.0, 0.37765, -0.16], [0.0, -0.37765, -0.16]],
        "thruster_directions": [(1.0, 1.0, 0.0), (1.0, 1.0, 0.0)], # useless if holonomic=True
        "positive_only": [True, True],
        "has_rudder": False,
        "has_keel": False,
        "has_sail": False,
        })

SPEC_SAILBOAT = ({
        "type": "single_motor_sailboat",          # or "vap2", "vap4", "kingfisher", "single_motor", etc.
        "thruster_layout": ["symmetric", "symmetric"],
        "thruster_curve_scale": 1.,
        "thruster_curve_bias": 0.,
        "hydro_lateral_drag": 0.7,
        "num_thrusters": 2,
        "holonomic": [True, True],
        "thruster_positions": [[0.0, 0.37765, -0.16], [0.0, -0.37765, -0.16]],
        "thruster_directions": [(1.0, 1.0, 0.0), (1.0, 1.0, 0.0)], # useless if holonomic=True
        "positive_only": [True, True],
        "has_rudder": True,
        "has_keel": True,
        "has_sail": True,
        })

SPEC_VAP4 = ({
        "type": "vap4",          # or "vap2", "vap4", "kingfisher", "single_motor", etc.
        "thruster_layout": ["asymmetric", "asymmetric", "asymmetric", "asymmetric"],
        "thruster_curve_scale": 1.,
        "thruster_curve_bias": 0.,
        "hydro_lateral_drag": 0.7,
        "num_thrusters": 4,
        "holonomic": [True, True, True, True],
        "thruster_positions": [[-0.53, 0.37765, -0.16], [-0.53, -0.37765, -0.16], [0.53, 0.37765, -0.16], [0.53, -0.37765, -0.16]],
        "thruster_directions": [(1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0)], # useless if holonomic=True
        "positive_only": [True, True, True, True],
        "has_rudder": False,
        "has_keel": False,
        "has_sail": False,
        })