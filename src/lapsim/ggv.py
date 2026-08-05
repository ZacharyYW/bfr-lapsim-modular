from dataclasses import dataclass, field


@dataclass
class GGV:
    lat_accels: dict[float, float] = field(default_factory=dict)
    long_accels: dict[float, float] = field(default_factory=dict)
    brake_accels: dict[float, float] = field(default_factory=dict)

    def populate_lat_accels(self) -> None:
        self.lat_accels = {r: 0.0 for r in [3.0, 6.0, 9.0, 12.0]}

    def populate_long_accels(self) -> None:
        self.long_accels = {v: 0.0 for v in [3.0, 6.0, 9.0, 12.0]}

    def populate_brake_accels(self) -> None:
        self.brake_accels = {v: 0.0 for v in [3.0, 6.0, 9.0, 12.0]}
