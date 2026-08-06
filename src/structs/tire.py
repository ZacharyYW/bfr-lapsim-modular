from dataclasses import dataclass, field
from pathlib import Path

from scipy.io import loadmat


@dataclass
class Tire:
    front_static_camber: float = -1.901  # deg
    rear_static_camber: float = -2.635   # deg

    # TODO: fill in real values -- placeholders below are 0.0 so the class
    # is at least constructible; using these for a real sim run will
    # silently zero out camber gain until they're set.
    camber_gain_roll: dict[str, float] = field(default_factory=lambda: {
        "FO": 0.0,
        "FI": 0.0,
        "RO": 0.0,
        "RI": 0.0,
    })
    camber_gain_pitch: dict[str, float] = field(default_factory=lambda: {
        "F": 0.0,
        "R": 0.0,
    })

    all_static_camber: dict[str, float] = field(init=False)

    # Pacejka BNP 1989 coefficients, a0-a13 (14 values), set via load_coefficients
    coeffs: dict[str, float] | None = field(default=None, init=False)
    coeff_source: str | None = field(default=None, init=False)
    coeff_rms_error_n: float | None = field(default=None, init=False)

    COEFF_NAMES = (
        "a0", "a1", "a2", "a3", "a4", "a5", "a6",
        "a7", "a8", "a9", "a10", "a11", "a12", "a13",
    )

    def load_coefficients(self, path: str | Path) -> None:
        """Load fitted Pacejka BNP 1989 coefficients from a .mat file produced
        by hoosier_r20_tire_model.m (the `tireParams` struct).

        Expects `tireParams.coeffs` to hold a0..a13 (14 values) and
        `tireParams.rms_error_N` for fit quality. Populates self.coeffs as a
        name -> value dict for easy access (e.g. self.coeffs["a3"]).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tire coefficient file not found: {path}")

        mat = loadmat(path, struct_as_record=False, squeeze_me=True)

        if "tireParams" not in mat:
            raise KeyError(
                f"Expected top-level variable 'tireParams' in {path}, "
                f"found: {list(mat.keys())}"
            )

        tire_params = mat["tireParams"]
        raw_coeffs = tire_params.coeffs

        if len(raw_coeffs) != len(self.COEFF_NAMES):
            raise ValueError(
                f"Expected {len(self.COEFF_NAMES)} coefficients "
                f"(a0-a{len(self.COEFF_NAMES) - 1}), got {len(raw_coeffs)} "
                f"from {path}. Check the MATLAB script's coeff_names still "
                f"matches the fit vector p_fit."
            )

        self.coeffs = dict(zip(self.COEFF_NAMES, raw_coeffs.tolist()))
        self.coeff_source = str(getattr(tire_params, "tire", path.name))
        self.coeff_rms_error_n = float(getattr(tire_params, "rms_error_N", float("nan")))