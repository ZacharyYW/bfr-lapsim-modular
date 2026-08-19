import numpy as np
import pprint
from scipy.io import loadmat
from scipy.optimize import minimize_scalar
from pathlib import Path
from tyres.pacejka_model_lat import pacejka_lat_force
from tyres.pacejka_model_long import pacejka_long_force
from tyres.pacejka_model_brake import pacejka_brake_force

G = 9.81  # m/s^2

TOTAL_CAR_MASS = 288
WHEELBASE = 1.53          # m
CG_HEIGHT = 0.30          # m
TIRE_RADIUS = 0.2032      # m
BRAKE_BIAS_FRONT = 0.70   # fraction of total braking force carried by front axle

TRACK_WIDTH = 1.245  # m, front == rear
CAMBER = 0.0         # rad — set nonzero if you want camber effects included

WD_FRONT_CHOICES = [0.52, 0.51, 0.5, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42]
WD_LAT_ACCEL_OUTPUTS = dict()
WD_LONG_ACCEL_OUTPUTS = dict()
WD_BRAKE_ACCEL_OUTPUTS = dict()

# For weight distribution of around 0.48/0.5
INITIAL_WD_FRONT = 0.50
INITIAL_LAT_ACCEL = 1.5
INITIAL_LONG_ACCEL = 1.2
INITIAL_BRAKE_ACCEL = 1.2

CONVERGENCE_TOL = 0.005

PACEJKA_BRAKE_COEFFS_PATH = Path("../data/coeffs/brake/[B2356run73] hoosier_r20_tire_params_brake.mat").expanduser().resolve()
PACEJKA_LONG_COEFFS_PATH = Path("../data/coeffs/longitudinal/[B2356run73] hoosier_r20_tire_params_long.mat").expanduser().resolve()
PACEJKA_LAT_COEFFS_PATH = Path("../data/coeffs/lateral/[B2356run9] hoosier_r20_tire_params_lat.mat").expanduser().resolve()

# For typical SA and SR values
LAT_ACCEL_SA = 2.75
LONG_ACCEL_SR = 0.15

_long_coeffs = None
_lat_coeffs = None


def load_pacejka_lat_coeffs():
    global _lat_coeffs
    mat = loadmat(PACEJKA_LAT_COEFFS_PATH)

    _lat_coeffs = np.asarray(mat["coeffs"]).flatten()  # expects a0..a13
    return _lat_coeffs


def load_pacejka_long_coeffs():
    global _long_coeffs
    mat = loadmat(PACEJKA_LONG_COEFFS_PATH)

    _long_coeffs = np.asarray(mat["coeffs"]).flatten()  # expects b0..b13
    return _long_coeffs


def load_pacejka_brake_coeffs():
    global _brake_coeffs
    mat = loadmat(PACEJKA_BRAKE_COEFFS_PATH)

    _brake_coeffs = np.asarray(mat["coeffs"]).flatten()  # expects b0..b13
    return _brake_coeffs


def _calculate_swd(wd):
    """Static wheel loads (N), 50/50 left-right split assumed."""
    total_weight = TOTAL_CAR_MASS * G
    front_axle_load = wd * total_weight
    rear_axle_load = (1 - wd) * total_weight
    return {
        "FO": front_axle_load / 2,
        "FI": front_axle_load / 2,
        "RO": rear_axle_load / 2,
        "RI": rear_axle_load / 2,
    }


# ---------------------------------------------------------------------
# Lateral — stubbed pending track width (front/rear)
# ---------------------------------------------------------------------

def generate_lat_accels():
    for curr_wd in WD_FRONT_CHOICES:
        swd_loads = _calculate_swd(curr_wd)

        lat_accel_guess = INITIAL_LAT_ACCEL
        next_lat_accel_guess = _calculate_lat_accel(lat_accel_guess, swd_loads)

        while abs(lat_accel_guess - next_lat_accel_guess) > CONVERGENCE_TOL:
            lat_accel_guess = next_lat_accel_guess
            next_lat_accel_guess = _calculate_lat_accel(lat_accel_guess, swd_loads)

        WD_LAT_ACCEL_OUTPUTS[curr_wd] = next_lat_accel_guess


def _calculate_lat_accel(lat_accel_guess, swd_loads, slip_angle=LAT_ACCEL_SA):
    """
    lat_accel_guess is in g's. Cornering transfers load from inner
    to outer wheels on BOTH axles (front and rear both engage,
    unlike the RWD-only longitudinal case).
    """
    ay = lat_accel_guess * G  # m/s^2
    weight_transfer = (TOTAL_CAR_MASS * ay * CG_HEIGHT) / TRACK_WIDTH  # N, per-axle

    front_axle_static = swd_loads["FO"] + swd_loads["FI"]
    rear_axle_static = swd_loads["RO"] + swd_loads["RI"]

    fo_load = front_axle_static / 2 + weight_transfer / 2
    fi_load = front_axle_static / 2 - weight_transfer / 2
    ro_load = rear_axle_static / 2 + weight_transfer / 2
    ri_load = rear_axle_static / 2 - weight_transfer / 2

    # clip in case weight transfer exceeds static load (inside wheel lifting)
    fi_load = max(fi_load, 0.0)
    ri_load = max(ri_load, 0.0)

    fy_fo = pacejka_lat_force(_lat_coeffs, slip_angle, fo_load)
    fy_fi = pacejka_lat_force(_lat_coeffs, slip_angle, fi_load)
    fy_ro = pacejka_lat_force(_lat_coeffs, slip_angle, ro_load)
    fy_ri = pacejka_lat_force(_lat_coeffs, slip_angle, ri_load)

    total_lateral_force = fy_fo + fy_fi + fy_ro + fy_ri
    new_lat_accel = total_lateral_force / (TOTAL_CAR_MASS * G)  # back to g's

    return new_lat_accel

def scale_lat_accels():
    max_lat_accel_even_dist = WD_LAT_ACCEL_OUTPUTS[INITIAL_WD_FRONT]
    lat_accel_scaling_factor = INITIAL_LAT_ACCEL / max_lat_accel_even_dist

    for wd in WD_LAT_ACCEL_OUTPUTS:
        WD_LAT_ACCEL_OUTPUTS[wd] *= lat_accel_scaling_factor

# ---------------------------------------------------------------------
# Longitudinal — traction-limited, RWD
# ---------------------------------------------------------------------

def generate_long_accels():
    for curr_wd in WD_FRONT_CHOICES:
        swd_loads = _calculate_swd(curr_wd)

        long_accel_guess = INITIAL_LONG_ACCEL
        next_long_accel_guess = _calculate_long_accel(long_accel_guess, swd_loads)

        while abs(long_accel_guess - next_long_accel_guess) > CONVERGENCE_TOL:
            long_accel_guess = next_long_accel_guess
            next_long_accel_guess = _calculate_long_accel(long_accel_guess, swd_loads)

        WD_LONG_ACCEL_OUTPUTS[curr_wd] = next_long_accel_guess


def _calculate_long_accel(long_accel_guess, swd_loads):
    """
    long_accel_guess is in g's. RWD -> only rear corners produce
    tractive force. Weight transfers rearward under acceleration.
    """
    ax = long_accel_guess * G  # m/s^2
    weight_transfer = (TOTAL_CAR_MASS * ax * CG_HEIGHT) / WHEELBASE  # N, axle-level

    # RWD: acceleration moves load onto the rear, off the front
    rear_axle_static = swd_loads["RO"] + swd_loads["RI"]
    rear_axle_dynamic = rear_axle_static + weight_transfer

    # assume even left/right split (no lateral component in pure long accel)
    rear_corner_load = rear_axle_dynamic / 2

    fx_ro = pacejka_long_force(_long_coeffs, LONG_ACCEL_SR, rear_corner_load)
    fx_ri = pacejka_long_force(_long_coeffs, LONG_ACCEL_SR, rear_corner_load)

    total_tractive_force = fx_ro + fx_ri
    new_long_accel = total_tractive_force / (TOTAL_CAR_MASS * G)  # back to g's

    return new_long_accel

def scale_long_accels():
    max_long_accel_even_dist = WD_LONG_ACCEL_OUTPUTS[INITIAL_WD_FRONT]
    long_accel_scaling_factor = INITIAL_LONG_ACCEL / max_long_accel_even_dist

    for wd in WD_LONG_ACCEL_OUTPUTS:
        WD_LONG_ACCEL_OUTPUTS[wd] *= long_accel_scaling_factor

# ---------------------------------------------------------------------
# Braking — traction-limited, fixed 0.7/0.3 bias
# ---------------------------------------------------------------------

def generate_brake_accels():
    for curr_wd in WD_FRONT_CHOICES:
        swd_loads = _calculate_swd(curr_wd)

        brake_accel_guess = INITIAL_BRAKE_ACCEL
        next_brake_accel_guess = _calculate_brake_accel(brake_accel_guess, swd_loads)

        while abs(brake_accel_guess - next_brake_accel_guess) > CONVERGENCE_TOL:
            brake_accel_guess = next_brake_accel_guess
            next_brake_accel_guess = _calculate_brake_accel(brake_accel_guess, swd_loads)

        WD_BRAKE_ACCEL_OUTPUTS[curr_wd] = next_brake_accel_guess


def _calculate_brake_accel(brake_accel_guess, swd_loads):
    """
    brake_accel_guess is a positive magnitude in g's.
    Braking moves load onto the front, off the rear.
    Bias is FIXED (0.7 front / 0.3 rear) -- whichever axle hits its
    friction limit first sets the achievable deceleration.
    """
    ax = brake_accel_guess * G
    weight_transfer = (TOTAL_CAR_MASS * ax * CG_HEIGHT) / WHEELBASE

    front_axle_static = swd_loads["FO"] + swd_loads["FI"]
    rear_axle_static = swd_loads["RO"] + swd_loads["RI"]

    front_axle_dynamic = front_axle_static + weight_transfer
    rear_axle_dynamic = rear_axle_static - weight_transfer

    front_corner_load = front_axle_dynamic / 2
    rear_corner_load = rear_axle_dynamic / 2

    fx_fo = pacejka_brake_force( _long_coeffs, front_corner_load, LONG_ACCEL_SR,)
    fx_fi = pacejka_brake_force( _long_coeffs, front_corner_load, LONG_ACCEL_SR)
    fx_ro = pacejka_brake_force( _long_coeffs, rear_corner_load, LONG_ACCEL_SR)
    fx_ri = pacejka_brake_force( _long_coeffs, rear_corner_load, LONG_ACCEL_SR)

    max_front_force = fx_fo + fx_fi
    max_rear_force = fx_ro + fx_ri

    # total achievable braking force, given the fixed bias, is capped
    # by whichever axle saturates first
    total_force_front_limited = max_front_force / BRAKE_BIAS_FRONT
    total_force_rear_limited = max_rear_force / (1 - BRAKE_BIAS_FRONT)

    total_braking_force = min(total_force_front_limited, total_force_rear_limited)
    new_brake_accel = total_braking_force / (TOTAL_CAR_MASS * G)

    return new_brake_accel

def scale_brake_accels():
    max_brake_accel_even_dist = WD_BRAKE_ACCEL_OUTPUTS[INITIAL_WD_FRONT]
    brake_accel_scaling_factor = INITIAL_BRAKE_ACCEL / max_brake_accel_even_dist

    for wd in WD_BRAKE_ACCEL_OUTPUTS:
        WD_BRAKE_ACCEL_OUTPUTS[wd] *= brake_accel_scaling_factor

# ---------- 

def calculate_tire_scaling_factor():
    selected_corner_radii = [17.2, 12.5, 24.3]
    selected_lat_accel = [1.5, 1.3, 1.5] # In g's

    even_wd_loads = _calculate_swd(0.5)

    for idx in range(len(selected_corner_radii)):
        curr_radii = selected_corner_radii[idx]
        curr_lat_accel = selected_lat_accel[idx]

        slip_angle = 12
        curr_yaw_moment = -1
        diff = -1

        final_lat_accel = -1

        while diff == -1 or diff > 0.025:
            calculated_lat_accel = _calculate_lat_accel(curr_lat_accel, even_wd_loads, slip_angle)

            final_lat_accel = calculated_lat_accel
            diff = abs(abs(calculated_lat_accel) - abs(curr_lat_accel))

            slip_angle -= 0.25

        print("Converged steady-state slip angle: ", slip_angle)
        print("converged lat accel: ", final_lat_accel)





if __name__ == "__main__":
    load_pacejka_lat_coeffs()
    load_pacejka_long_coeffs()
    load_pacejka_brake_coeffs()

    calculate_tire_scaling_factor()

    generate_long_accels()
    scale_long_accels()

    generate_lat_accels()
    scale_lat_accels()

    generate_brake_accels()
    scale_brake_accels()

    pprint.pprint(WD_LONG_ACCEL_OUTPUTS, width=40)
    pprint.pprint(WD_LAT_ACCEL_OUTPUTS, width=40)
    pprint.pprint(WD_BRAKE_ACCEL_OUTPUTS, width=40)