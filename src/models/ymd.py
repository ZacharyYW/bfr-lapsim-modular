import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path

PACEJKA_LAT_COEFFS_PATH = Path("../data/coeffs/lateral/[B2356run9] hoosier_r20_tire_params_lat.mat").expanduser().resolve()

LLTD = 0.50  # LLTD, perfect split between front and rear
SKIDPAD_TIME = 5.6  # Time in seconds
SKIDPAD_RADIUS = 8.3975  # Radius in meters

MAX_STEERING_ANGLE = 20  # Max steering angle, in degrees

# ---------------------------------------------------------------------------
# Vehicle parameters — replace with your car's actual numbers
# ---------------------------------------------------------------------------
MASS = 268.0              # kg, total vehicle mass
weight_dist_front = 0.50  # fraction of static weight on the front axle
WHEELBASE = 1.53          # m
TRACK_FRONT = 1.245        # m
TRACK_REAR = 1.245         # m
CG_HEIGHT = 0.30          # m
G = 9.81

AERO_CLA = 2.84

# a = WHEELBASE * (1 - weight_dist_front)  # CG -> front axle distance
# b = WHEELBASE * weight_dist_front        # CG -> rear axle distance
a = None
b = None

# Sweep resolution
N_BETA = 241
N_DELTA = 241
BETA_LIMIT_DEG = 20  # vehicle slip angle sweep range, tune to your car

BETA_RANGE = np.linspace(-BETA_LIMIT_DEG, BETA_LIMIT_DEG, N_BETA)
DELTA_RANGE = np.linspace(-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE, N_DELTA)

yaw_rate = None
velocity_long = None
desired_lat_accel = None
lat_coeffs = None

def _setup():
    global yaw_rate, velocity_long, desired_lat_accel
    yaw_rate = (2 * math.pi) / SKIDPAD_TIME
    velocity_long = (2 * math.pi * SKIDPAD_RADIUS) / SKIDPAD_TIME

    # desired_lat_accel = (velocity_long) ** 2 / SKIDPAD_RADIUS
    desired_lat_accel = 10.57


# ---------------------------------------------------------------------------
# Tire model interface — plug your existing model in here.
# Sign convention: positive alpha -> positive (this axle's) Fy in the
# direction that reduces alpha (standard SAE-ish tire convention). Make sure
# your model's sign convention matches what's used below, or flip signs.
# ---------------------------------------------------------------------------

def load_coeffs():
    global lat_coeffs
    mat = loadmat(PACEJKA_LAT_COEFFS_PATH)

    lat_coeffs = np.asarray(mat["coeffs"]).flatten()  # expects a0..a13
    return lat_coeffs


def pacejka_lat_force(p, alpha, Fz, gamma_deg=0.0):
    """
    alpha : slip angle [deg]
    Fz    : normal load [N] (positive, compressive)
    returns Fy [N]
    """
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 = p

    C = a0
    D = a1 * Fz**2 + a2 * Fz
    BCD = a3 * np.sin(2 * np.arctan(Fz / a4)) * (1 - a5 * np.abs(gamma_deg))
    B = np.divide(BCD, C * D, out=np.zeros_like(np.asarray(BCD, dtype=float)), where=(C * D) != 0)
    E = a6 * Fz + a7
    SH = a8 * gamma_deg + a9 * Fz + a10
    SV = a11 * Fz * gamma_deg + a12 * Fz + a13

    phi = alpha + SH
    FY = D * np.sin(C * np.arctan(B * phi - E * (B * phi - np.arctan(B * phi)))) + SV
    return (FY * 0.7)


def downforce(V):
    return 0.5 * 1.225 * (V ** 2)


def corner_loads(Ay):
    """Static load + simplistic lateral load transfer split via LLTD."""
    Fz_static_front = MASS * G * weight_dist_front / 2
    Fz_static_rear = MASS * G * (1 - weight_dist_front) / 2

    track_avg = (TRACK_FRONT + TRACK_REAR) / 2
    dFz_total = MASS * Ay * CG_HEIGHT / track_avg
    dFz_front = dFz_total * LLTD
    dFz_rear = dFz_total * (1 - LLTD)

    FzFL = max(Fz_static_front - dFz_front / 2, 0.0)
    FzFR = max(Fz_static_front + dFz_front / 2, 0.0)
    FzRL = max(Fz_static_rear - dFz_rear / 2, 0.0)
    FzRR = max(Fz_static_rear + dFz_rear / 2, 0.0)
    return FzFL, FzFR, FzRL, FzRR


# def solve_point_constant_radius(beta, delta, R, max_iter=50, tol=1e-6):
#     r = 0.0
#     Ay = 0.0
#     Fy_front = Fy_rear = 0.0

#     alpha_f_deg = 0.0
#     alpha_r_deg = 0.0

#     for _ in range(max_iter):
#         alpha_f = math.radians(beta) + (a * r / V) - math.radians(delta)
#         alpha_r = math.radians(beta) - (b * r / V)

#         alpha_f_deg = math.degrees(alpha_f)
#         alpha_r_deg = math.degrees(alpha_r)

#         FzFL, FzFR, FzRL, FzRR = corner_loads(Ay)
#         FzDF = downforce(V) / 4
#         # print("Generated Corner Loads: ", FzFL, FzFR, FzRL, FzRR)

#         FyFL = pacejka_lat_force(lat_coeffs, alpha_f_deg, FzFL + FzDF)
#         FyFR = pacejka_lat_force(lat_coeffs, alpha_f_deg, FzFR + FzDF)
#         FyRL = pacejka_lat_force(lat_coeffs, alpha_r_deg, FzRL + FzDF)
#         FyRR = pacejka_lat_force(lat_coeffs, alpha_r_deg, FzRR + FzDF)

#         Fy_front = FyFL + FyFR
#         Fy_rear = FyRL + FyRR

#         Ay_new = (Fy_front + Fy_rear) / MASS
#         r_new = Ay_new / V

#         if abs(Ay_new - Ay) < tol and abs(r_new - r) < tol:
#             Ay, r = Ay_new, r_new
#             break
#         Ay, r = Ay_new, r_new

#     Mz = a * Fy_front - b * Fy_rear
#     return Ay, Mz, r, alpha_f_deg, alpha_r_deg


def solve_point(beta, delta, V, max_iter=50, tol=1e-6):
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

    alpha_f_deg = 0.0
    alpha_r_deg = 0.0

    for _ in range(max_iter):
        alpha_f = math.radians(beta) + (a * r / V) - math.radians(delta)
        alpha_r = math.radians(beta) - (b * r / V)

        alpha_f_deg = math.degrees(alpha_f)
        alpha_r_deg = math.degrees(alpha_r)

        FzFL, FzFR, FzRL, FzRR = corner_loads(Ay)
        FzDF = downforce(V) / 4
        # print("Generated Corner Loads: ", FzFL, FzFR, FzRL, FzRR)

        FyFL = pacejka_lat_force(lat_coeffs, alpha_f_deg, FzFL + FzDF)
        FyFR = pacejka_lat_force(lat_coeffs, alpha_f_deg, FzFR + FzDF)
        FyRL = pacejka_lat_force(lat_coeffs, alpha_r_deg, FzRL + FzDF)
        FyRR = pacejka_lat_force(lat_coeffs, alpha_r_deg, FzRR + FzDF)

        Fy_front = FyFL + FyFR
        Fy_rear = FyRL + FyRR

        Ay_new = (Fy_front + Fy_rear) / MASS
        r_new = Ay_new / V

        if abs(Ay_new - Ay) < tol and abs(r_new - r) < tol:
            Ay, r = Ay_new, r_new
            break
        Ay, r = Ay_new, r_new

    Mz = a * Fy_front - b * Fy_rear
    return Ay, Mz, r, alpha_f_deg, alpha_r_deg


def build_ymd(V):
    """Sweep beta x delta and return grids of Ay, Mz (shape [beta, delta])."""
    Ay_grid = np.zeros((len(BETA_RANGE), len(DELTA_RANGE)))
    Mz_grid = np.zeros_like(Ay_grid)

    min_beta = math.degrees(math.atan(weight_dist_front * WHEELBASE / SKIDPAD_RADIUS))
    # 11.351 deg
    desired_delta = 11.351

    for i, beta in enumerate(BETA_RANGE):
        for j, delta in enumerate(DELTA_RANGE):
            Ay, Mz, r, alpha_f, alpha_r = solve_point(beta, delta, V)
            # Current Skidpad Filter

            # if abs(abs(Ay) - abs(desired_lat_accel)) <= 0.25:
            #     print("Found this Ay: ", Ay)
            #     print("Found this Mz: ", Mz)
            #     print("Found this Steering Angle: ", delta)
            #     print("Found this Body Slip Angle: ", beta)
            #     print("Found this yaw velocity: ", r)
            #     print("Found this af: ", alpha_f)
            #     print("Found this ar: ", alpha_r)
            #     print()
            Ay_grid[i, j] = Ay
            Mz_grid[i, j] = Mz

    print("Max Lateral Accel: ", np.max(Ay_grid))

    return Ay_grid, Mz_grid


def plot_ymd(Ay_grid, Mz_grid):
    fig, ax = plt.subplots(figsize=(12, 12))

    # constant-beta lines: fix beta (row), sweep delta across columns
    for i in range(Ay_grid.shape[0]):
        ax.plot(Ay_grid[i, :], Mz_grid[i, :], color="tab:blue", lw=0.75)

    # constant-delta lines: fix delta (column), sweep beta across rows
    for j in range(Ay_grid.shape[1]):
        ax.plot(Ay_grid[:, j], Mz_grid[:, j], color="tab:red", lw=0.75)

    ax.axhline(0, color="k", lw=0.75)
    ax.axvline(0, color="k", lw=0.75)
    ax.set_xlabel("Lateral Acceleration Ay [m/s^2]", fontsize=20)
    ax.set_ylabel("Yaw Moment Mz [Nm]", fontsize=20)
    ax.set_title("Yaw Moment Diagram (FSAE Michigan 2026 Skidpad)", fontsize=20)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(Path(f"../figures/ymd.png").expanduser().resolve())


def extract_trim_locus(Ay_grid, Mz_grid, beta_range_deg, delta_range_deg):
    """
    Scan each delta column for Mz=0 crossings across the beta sweep.
    Returns a list of dicts, one per crossing found (a column can have
    more than one crossing if Mz isn't monotonic in beta):

        delta_deg   - steer angle for this column
        beta_trim   - interpolated body slip angle at the crossing
        Ay_trim     - interpolated lateral acceleration at the crossing
        dMz_dbeta   - local slope of Mz vs beta at the crossing [N*m/deg]
        stable      - True if dMz_dbeta < 0 (restoring moment)
    """
    n_beta, n_delta = Mz_grid.shape
    trim_locus = []

    for j in range(n_beta):
        mz_col = Mz_grid[j, :]
        ay_col = Ay_grid[j, :]

        for i in range(n_delta - 1):
            mz0, mz1 = mz_col[i], mz_col[i + 1]

            # CHecks to make sure the pair of points either: 1. Starts at Mz = 0 (hovering around steady-state) or 2. Transitions Yaw Moment signs (hovering around steady-state)
            if mz0 == 0.0 or mz0 * mz1 < 0:
                if mz0 == 0.0:
                    frac = 0.0
                else:
                    frac = -mz0 / (mz1 - mz0)  # linear interpolation fraction

                # beta_trim = beta_range_deg[i] + frac * (beta_range_deg[i + 1] - beta_range_deg[i])
                delta_trim = delta_range_deg[i] + frac * (delta_range_deg[i + 1] - delta_range_deg[i])

                Ay_trim = ay_col[i] + frac * (ay_col[i + 1] - ay_col[i])
                # dMz_dbeta = (mz1 - mz0) / (beta_range_deg[i + 1] - beta_range_deg[i])
                dMz_ddelta = (mz1 - mz0) / (delta_range_deg[i+1] - delta_range_deg[i])

                trim_locus.append({
                    "delta_deg": delta_range_deg[j],
                    "beta_trim": delta_trim,
                    "Ay_trim": Ay_trim,
                    "dMz_dbeta": 0,
                    "dMz_ddelta": dMz_ddelta,
                    "stable": dMz_ddelta < 0,
                })

    return trim_locus


def find_trim_at_ay(trim_locus, target_ay):
    """Return the trim-locus point whose Ay is closest to target_ay."""
    if not trim_locus:
        return None
    return min(trim_locus, key=lambda pt: abs(pt["Ay_trim"] - target_ay))

def main():
    _setup()
    load_coeffs()

    fig, ax = plt.subplots(figsize=(6, 6))

    weight_dist_points = []
    dMz_ddelta_points = []

    for curr_weight_dist_front in [0.55, 0.54, 0.53, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47, 0.46, 0.45]:
        global a, b, weight_dist_front

        weight_dist_front = curr_weight_dist_front

        a = WHEELBASE * (1 - weight_dist_front)  # CG -> front axle distance
        b = WHEELBASE * weight_dist_front        # CG -> rear axle distance

        Ay_grid, Mz_grid = build_ymd(velocity_long)

        trim_locus = extract_trim_locus(Ay_grid, Mz_grid, BETA_RANGE, DELTA_RANGE)
        skidpad_ay_target = velocity_long**2 / SKIDPAD_RADIUS  # ~10.57 m/s^2

        trim_point = find_trim_at_ay(trim_locus, skidpad_ay_target)
        if trim_point:
            tendency = "understeer (stable)" if trim_point["stable"] else "oversteer (unstable)"
            print(f"Skidpad trim point (target Ay={skidpad_ay_target:.3f}):")
            print(f"  delta   = {trim_point['delta_deg']:.3f} deg")
            print(f"  beta    = {trim_point['beta_trim']:.3f} deg")
            print(f"  Ay      = {trim_point['Ay_trim']:.3f} m/s^2")
            print(f"  dMz/dβ  = {trim_point['dMz_dbeta']:.3f} N*m/deg  -> {tendency}")
            print(f"  dMz/ddelta = {trim_point['dMz_ddelta']:.3f} N*m/deg  -> {tendency}")

            weight_dist_points.append(curr_weight_dist_front)
            dMz_ddelta_points.append(trim_point['dMz_ddelta'])

            ax.scatter(curr_weight_dist_front, trim_point['dMz_ddelta'], color="red", zorder=3)
        else:
            print("No trim (Mz=0) crossings found — widen your beta/delta sweep range.")

        plot_ymd(Ay_grid, Mz_grid)

    weight_dist_arr = np.array(weight_dist_points)
    dMz_ddelta_arr = np.array(dMz_ddelta_points)

    # --- Curve of best fit ---
    if len(weight_dist_arr) >= 3:
        FIT_DEGREE = 2  # bump to 1 for a straight-line fit, or higher if the trend is more complex
        coeffs = np.polyfit(weight_dist_arr, dMz_ddelta_arr, FIT_DEGREE)
        fit_fn = np.poly1d(coeffs)

        x_fit = np.linspace(weight_dist_arr.min(), weight_dist_arr.max(), 200)
        y_fit = fit_fn(x_fit)

        ax.plot(x_fit, y_fit, color="tab:blue", lw=2, zorder=2,
                label=f"deg-{FIT_DEGREE} fit")
        ax.legend(fontsize=14)

    # --- Autoscale to the relevant data range, with a little padding ---
    if len(weight_dist_arr) > 0:
        x_pad = (weight_dist_arr.max() - weight_dist_arr.min()) * 0.1 or 0.01
        y_pad = (dMz_ddelta_arr.max() - dMz_ddelta_arr.min()) * 0.1 or 1.0

        ax.set_xlim(weight_dist_arr.min() - x_pad + 0.01, weight_dist_arr.max() + x_pad - 0.01)
        ax.set_ylim(dMz_ddelta_arr.min() - y_pad + 0.01, dMz_ddelta_arr.max() + y_pad - 0.01)

    ax.axhline(0, color="k", lw=0.75)
    ax.axvline(0, color="k", lw=0.75)
    ax.set_xlabel("Weight Distribution Front", fontsize=20)
    ax.set_ylabel("dMz/ddelta at Trim Line (Nm/deg)", fontsize=20)
    ax.set_title("Controllability Measure VS Weight Distribution Front", fontsize=20)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(Path(f"../figures/wd-analysis-graph.png").expanduser().resolve(), bbox_inches='tight')


if __name__ == "__main__":
    main()