from dataclasses import dataclass, field

from .aero import Aero
from .engine import Engine
from .suspension import Suspension


@dataclass
class Car:
    sus_params: Suspension = field(default_factory=Suspension)
    aero_params: Aero = field(default_factory=Aero)
    engine_params: Engine = field(default_factory=Engine)
    tire_params: dict | None = None

    def load_transfer(self, accel: float, vel: float) -> dict[str, float]:
        """Placeholder for wheel-load calculation."""
        return {
            "FO": 1000.0,
            "FI": 1000.0,
            "RO": 1000.0,
            "RI": 1000.0,
        }

    def cornering_force(self, *args, **kwargs) -> tuple[float, float]:
        return 0.0, 0.0

    def longitudinal_force(self, *args, **kwargs) -> tuple[float, float]:
        return 0.0, 0.0
