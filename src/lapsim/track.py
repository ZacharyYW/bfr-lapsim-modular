from dataclasses import dataclass, field


@dataclass
class Track:
    lengths: list[float] = field(default_factory=list)
    curvatures: list[float] = field(default_factory=list)

    def load(self, path: str) -> None:
        self.lengths = []
        self.curvatures = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    self.lengths.append(float(parts[0]))
                    self.curvatures.append(float(parts[1]))
