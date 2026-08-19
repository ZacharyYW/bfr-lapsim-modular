import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path

from structs.car import Car
from models.tire import Tire
from utils.constants import G
from utils.loads import calculate_corner_loads, calculate_downforce

class YMD:

    def __init__(self):
        self.car = Car()
        self.tire = Tire()

        # Variables store angles in degrees; computations on these angles involve changing them to radians first
        self.n_beta = 241
        self.n_delta = 241
        self.beta_limit = 20
        self.delta_limit = 20

        self.beta_range = np.linspace(-self.beta_limit, self.beta_limit, self.n_beta)
        self.delta_range = np.linspace(-self.delta_limit, self.delta_limit, self.n_delta)


    def _solve_point(self, beta, delta, V, max_iter=50, tol=1e-6):
        """
        Given body slip angle beta, steer angle delta, and speed V, iterate
        on yaw rate r (via r = Ay / V) until self-consistent, then return
        the resulting (Ay, Mz).

        If you'd rather skip the feedback loop (faster, less accurate near
        the limit), just set r = 0 and remove the loop — see note below.
        """
        r = 0.0
        Ay = 0.0
        Fy_front = Fy_rear = 0.0

        sus_params = self.car.sus_params

        a = (1 - sus_params.weight_dist_front) * sus_params.wheelbase
        b = (sus_params.weight_dist_front) * sus_params.wheelbase

        for _ in range(max_iter):
            alpha_f = math.degrees(math.radians(beta) + (a * r / V) - math.radians(delta))
            alpha_r = math.degrees(math.radians(beta) - (b * r / V))

            FzFL, FzFR, FzRL, FzRR = calculate_corner_loads(Ay)
            FzDF = calculate_downforce(V) / 4
            # print("Generated Corner Loads: ", FzFL, FzFR, FzRL, FzRR)

            # TODO: Calculate roll amount based on roll gradient OR roll rate
            FyFL = self.tire.calculate_lat_force(alpha_f, FzFL + FzDF, 0)
            FyFR = self.tire.calculate_lat_force(alpha_f, FzFR + FzDF, 0)
            FyRL = self.tire.calculate_lat_force(alpha_r, FzRL + FzDF, 0)
            FyRR = self.tire.calculate_lat_force(alpha_r, FzRR + FzDF, 0)

            Fy_front = FyFL + FyFR
            Fy_rear = FyRL + FyRR

            Ay_new = (Fy_front + Fy_rear) / sus_params.mass
            r_new = Ay_new / V

            if abs(Ay_new - Ay) < tol and abs(r_new - r) < tol:
                Ay, r = Ay_new, r_new
                break
            Ay, r = Ay_new, r_new

        Mz = a * Fy_front - b * Fy_rear
        return Ay, Mz, r, alpha_f, alpha_r


    def build_ymd(self, V):
        """Sweep beta x delta and return grids of Ay, Mz (shape [beta, delta])."""
        Ay_grid = np.zeros((len(self.beta_range), len(self.delta_range)))
        Mz_grid = np.zeros_like(Ay_grid)

        for i, beta in enumerate(self.beta_range):
            for j, delta in enumerate(self.delta_range):
                Ay, Mz, r, alpha_f, alpha_r = self._solve_point(beta, delta, V)

                Ay_grid[i, j] = Ay
                Mz_grid[i, j] = Mz

        print("Max Lateral Accel: ", np.max(Ay_grid))

        self.Ay_grid = Ay_grid
        self.Mz_grid = Mz_grid

        return Ay_grid, Mz_grid


    def plot_ymd(self, title, Ay_grid=None, Mz_grid=None):
        if not self.Ay_grid or not self.Mz_grid:
            print("No YMD built yet.")
            return

        fig, ax = plt.subplots(figsize=(16, 16))

        # constant-beta lines: fix beta (row), sweep delta across columns
        for i in range(self.Ay_grid.shape[0]):
            ax.plot(self.Ay_grid[i, :], self.Mz_grid[i, :], color="tab:blue", lw=0.75)

        # constant-delta lines: fix delta (column), sweep beta across rows
        for j in range(self.Ay_grid.shape[1]):
            ax.plot(self.Ay_grid[:, j], self.Mz_grid[:, j], color="tab:red", lw=0.75)

        ax.axhline(0, color="k", lw=0.75)
        ax.axvline(0, color="k", lw=0.75)
        ax.set_xlabel("Lateral Acceleration Ay [m/s^2]", fontsize=20)
        ax.set_ylabel("Yaw Moment Mz [Nm]", fontsize=20)
        ax.set_title(title, fontsize=20)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(Path(f"../figures/{title}.png").expanduser().resolve())

    def extract_ymd_gradient(self):
        if not self.Ay_grid or not self.Mz_grid:
            print("No YMD built yet.")
            return

        self.Mz_gradient_grid = np.gradient(self.Mz_grid)
        return self.Mz_gradient_grid

    # def extract_trim_locus(self, Ay_grid, Mz_grid):
    #     """
    #     Scan each delta column for Mz=0 crossings across the beta sweep.
    #     Returns a list of dicts, one per crossing found (a column can have
    #     more than one crossing if Mz isn't monotonic in beta):

    #         delta_deg   - steer angle for this column
    #         beta_trim   - interpolated body slip angle at the crossing
    #         Ay_trim     - interpolated lateral acceleration at the crossing
    #         dMz_dbeta   - local slope of Mz vs beta at the crossing [N*m/deg]
    #         stable      - True if dMz_dbeta < 0 (restoring moment)
    #     """
    #     n_beta, n_delta = Mz_grid.shape
    #     trim_locus = []

    #     for j in range(n_beta):
    #         mz_col = Mz_grid[j, :]
    #         ay_col = Ay_grid[j, :]

    #         for i in range(n_delta - 1):
    #             mz0, mz1 = mz_col[i], mz_col[i + 1]

    #             # CHecks to make sure the pair of points either: 1. Starts at Mz = 0 (hovering around steady-state) or 2. Transitions Yaw Moment signs (hovering around steady-state)
    #             if mz0 == 0.0 or mz0 * mz1 < 0:
    #                 if mz0 == 0.0:
    #                     frac = 0.0
    #                 else:
    #                     frac = -mz0 / (mz1 - mz0)  # linear interpolation fraction

    #                 # beta_trim = beta_range_deg[i] + frac * (beta_range_deg[i + 1] - beta_range_deg[i])
    #                 delta_trim = delta_range_deg[i] + frac * (delta_range_deg[i + 1] - delta_range_deg[i])

    #                 Ay_trim = ay_col[i] + frac * (ay_col[i + 1] - ay_col[i])
    #                 # dMz_dbeta = (mz1 - mz0) / (beta_range_deg[i + 1] - beta_range_deg[i])
    #                 dMz_ddelta = (mz1 - mz0) / (delta_range_deg[i+1] - delta_range_deg[i])

    #                 trim_locus.append({
    #                     "delta_deg": delta_range_deg[j],
    #                     "beta_trim": delta_trim,
    #                     "Ay_trim": Ay_trim,
    #                     "dMz_dbeta": 0,
    #                     "dMz_ddelta": dMz_ddelta,
    #                     "stable": dMz_ddelta < 0,
    #                 })

    #     return trim_locus


    # def find_trim_at_ay(trim_locus, target_ay):
    #     """Return the trim-locus point whose Ay is closest to target_ay."""
    #     if not trim_locus:
    #         return None
    #     return min(trim_locus, key=lambda pt: abs(pt["Ay_trim"] - target_ay))


if __name__ == "__main__":
    skidpad_time = 5.6
    skidpad_radius = 8.3975
    V = (2 * math.pi * skidpad_radius) / skidpad_time

    ymd = YMD()
    ymd.build_ymd(V)
    ymd.plot_ymd("FSAE Michigan 2026 Skidpad YMD")