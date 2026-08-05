from dataclasses import dataclass


@dataclass
class Suspension:
    mass: float = 268.0
    cg_height: float = 0.303
    trackwidth: float = 1.245
    wheelbase: float = 1.53
    weight_dist_front: float = 0.5
    lateral_load_transfer_dist: float = 0.5
    front_roll_stiffness: float = 40000.0
    pitch_stiffness: float = 0.0
    camber_gain: float = 0.0
