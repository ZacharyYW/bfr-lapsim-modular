from dataclasses import dataclass, field

from structs.aero import Aero
from structs.engine import Engine
from structs.suspension import Suspension

from utils.constants import AIR_DENSITY, G

import math

@dataclass
class Car:
    sus_params: Suspension = field(default_factory=Suspension)
    aero_params: Aero = field(default_factory=Aero)
    engine_params: Engine = field(default_factory=Engine)
    tire_params: dict | None = None

    def __init__(self) -> None:
        front_axle_load = self.sus_params.mass * self.sus_params.weight_dist_front
        rear_axle_load = self.sus_params * (1 - self.sus_params.weight_dist_front)

        self.tire_params = {
            "FO": front_axle_load / 2,
            "FI": front_axle_load / 2,
            "RO": rear_axle_load / 2,
            "RI": rear_axle_load / 2 
        }

    def load_transfer(self, lat_accel: float, long_accel: float, vel: float) -> dict[str, float]:
        """Placeholder for wheel-load calculation."""
        # Step 1: Calculate downforce
        total_downforce = (0.5) * AIR_DENSITY * math.pow(vel, 2)
        single_tire_downforce = total_downforce / 4

        # Step 2: Calculate loads after cornering/accelerating
        lat_load_transfer = (self.sus_params.mass * lat_accel * G) * (self.sus_params.cg_height) / self.sus_params.trackwidth
        long_load_transfer = (self.sus_params.mass * long_accel * G) * (self.sus_params.cg_height) / self.sus_params.wheelbase

        return {
            "FO" : self.tire_params["FO"] + lat_load_transfer + single_tire_downforce,
            "FI" : self.tire_params["FI"] - lat_load_transfer + single_tire_downforce,
            "RO" : self.tire_params["RO"] + long_load_transfer + single_tire_downforce,
            "RI" : self.tire_params["RI"] - long_load_transfer + single_tire_downforce
        }
    

    def cornering_force(self, *args, **kwargs) -> tuple[float, float]:
        return 0.0, 0.0

    def longitudinal_force(self, *args, **kwargs) -> tuple[float, float]:
        return 0.0, 0.0
