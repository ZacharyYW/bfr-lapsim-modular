from dataclasses import dataclass


@dataclass
class Aero:
    cla: float = 2.7
    cda: float = 1.0
    cop: float = 0.5
