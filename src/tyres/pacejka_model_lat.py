import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.optimize import least_squares, differential_evolution
from pathlib import Path

TYRE_DATA_PATH = Path("../../data/tyres/RunData_Cornering_Matlab_SI_Round9").expanduser().resolve()  # folder containing the B2356run*.mat TTC run files

# tire filter — Hoosier 43075 16x7.5-10 R20, 8" rim
TIRE_ID_SUBSTRINGS = ["43075", "16x7.5", "8 inch rim"]

FZ_MIN, FZ_MAX = 100.0, 1300.0   # [N] valid load window
SA_MAX = 14.0                    # [deg] valid slip angle window
PRESSURE_TARGET = 83.0           # [kPa]
PRESSURE_TOL = 5.0               # [kPa]
DECIMATE = 3                     # keep every Nth sample after filtering

_data_cache = None
_coeffs_cache = None


def _mat_str(val):
    """Coerce a loadmat string field (char array / object array) to a plain str."""
    arr = np.asarray(val)
    return str(arr.item()).strip() if arr.size == 1 else str(arr).strip()


def load_data():
    """
    Load and filter all matching TTC cornering runs from TYRE_DATA_PATH.
    Returns a dict of 1D numpy arrays: SA [deg], FY [N], FZ [N, positive], IA [deg].
    """
    global _data_cache
    if _data_cache is not None:
        return _data_cache

    run_files = sorted(glob.glob(os.path.join(TYRE_DATA_PATH, "B2356run*.mat")))
    if not run_files:
        raise FileNotFoundError(f"no B2356run*.mat files found in '{TYRE_DATA_PATH}'")

    sa_list, fy_list, fz_list, ia_list, p_list = [], [], [], [], []

    print("loading hoosier 43075 16x7.5-10 r20 (8 inch rim) runs...")
    for fpath in run_files:
        # cheap check first: just the tire id
        info = loadmat(fpath, variable_names=["tireid"])
        tire_str = _mat_str(info["tireid"])
        if not all(sub in tire_str for sub in TIRE_ID_SUBSTRINGS):
            continue

        d = loadmat(fpath, variable_names=["SA", "FY", "FZ", "IA", "P"])
        sa_list.append(np.asarray(d["SA"]).squeeze())
        fy_list.append(np.asarray(d["FY"]).squeeze())
        fz_list.append(np.asarray(d["FZ"]).squeeze())
        ia_list.append(np.asarray(d["IA"]).squeeze())
        p_list.append(np.asarray(d["P"]).squeeze())
        print(f"  loaded: {os.path.basename(fpath)}  ({tire_str})")

    if not sa_list:
        raise RuntimeError(
            "no Hoosier 43075 16x7.5-10 R20 (8 inch rim) runs matched. "
            "check TYRE_DATA_PATH and the tire id filter."
        )

    SA_all = np.concatenate(sa_list)
    FY_all = np.concatenate(fy_list)
    FZ_all = np.concatenate(fz_list)
    IA_all = np.concatenate(ia_list)
    P_all = np.concatenate(p_list)

    print(f"total raw data points: {len(SA_all)}\n")

    FZ_pos = np.abs(FZ_all)  # TTC stores FZ negative

    valid = (
        (FZ_pos > FZ_MIN) & (FZ_pos < FZ_MAX)
        & (np.abs(SA_all) < SA_MAX)
        & (np.abs(P_all - PRESSURE_TARGET) < PRESSURE_TOL)
    )

    SA = SA_all[valid][::DECIMATE]
    FY = FY_all[valid][::DECIMATE]
    FZ = FZ_pos[valid][::DECIMATE]
    IA = IA_all[valid][::DECIMATE]

    print(f"data points after filtering: {valid.sum()}  "
          f"(using every {DECIMATE}-th -> {SA.size} points)\n")

    _data_cache = {"SA": SA, "FY": FY, "FZ": FZ, "IA": IA}
    return _data_cache


def _mf_lateral(p, alpha_deg, Fz, gamma_deg=0.0):
    """
    Pacejka BNP 1989 lateral magic formula.
    alpha_deg, gamma_deg in DEGREES (matches this fit's calibration). Fz in N.
    Vectorized: alpha_deg / Fz / gamma_deg may be arrays (broadcastable).
    """
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 = p

    C = a0
    D = a1 * Fz**2 + a2 * Fz
    BCD = a3 * np.sin(2 * np.arctan(Fz / a4)) * (1 - a5 * np.abs(gamma_deg))
    B = np.divide(BCD, C * D, out=np.zeros_like(np.asarray(BCD, dtype=float)), where=(C * D) != 0)
    E = a6 * Fz + a7
    SH = a8 * gamma_deg + a9 * Fz + a10
    SV = a11 * Fz * gamma_deg + a12 * Fz + a13

    phi = alpha_deg + SH
    FY = D * np.sin(C * np.arctan(B * phi - E * (B * phi - np.arctan(B * phi)))) + SV
    return FY


# def generate_coeffs(seed=0):
#     global _coeffs_cache
#     if _coeffs_cache is not None:
#         return _coeffs_cache

#     data = load_data()
#     SA, FY, FZ, IA = data["SA"], data["FY"], data["FZ"], data["IA"]

#     # ---- stage 1: anchor peak-force load curve (unchanged) ----
#     fz_bin_centers = np.arange(150, 1251, 50)
#     fz_bin_tol = 55.0
#     fzc, dpk = [], []
#     for fz in fz_bin_centers:
#         sel = (np.abs(FZ - fz) < fz_bin_tol) & (np.abs(SA) > 5) & (np.abs(IA) < 0.6)
#         if sel.sum() > 40:
#             v = np.sort(np.abs(FY[sel]))
#             fzc.append(fz)
#             dpk.append(v[max(0, round(0.90 * len(v)) - 1)])
#     fzc = np.asarray(fzc, dtype=float)
#     dpk = np.asarray(dpk, dtype=float)

#     A = np.column_stack([fzc**2, fzc])
#     ab, *_ = np.linalg.lstsq(A, -dpk, rcond=None)
#     a1_fix, a2_fix = ab
#     print(f"stage 1 — anchored load curve: a1 = {a1_fix:.4e}, a2 = {a2_fix:.4f}")

#     def expand(free):
#         a0, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 = free
#         return np.array([a0, a1_fix, a2_fix, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13])

#     def residuals(free):
#         p = expand(free)
#         model = _mf_lateral(p, SA, FZ, IA)
#         return (FY - model) / FZ

#     def cost_fn(free):
#         r = residuals(free)
#         return np.sum(r ** 2)

#     lb = np.array([1.2, -5000, 100, 0.0, -5e-4, -1.0, -2, -1e-2, -5, -10, -2, -200])
#     ub = np.array([1.8, 0, 5000, 0.5, 5e-4, 1.5, 2, 1e-2, 5, 10, 2, 200])

#     # ---- stage 2: global search with differential_evolution ----
#     print("stage 2 — global search (differential_evolution)...")
#     de_result = differential_evolution(
#         cost_fn,
#         bounds=list(zip(lb, ub)),
#         seed=seed,
#         maxiter=300,
#         popsize=25,
#         tol=1e-9,
#         mutation=(0.5, 1.5),
#         recombination=0.7,
#         polish=False,   # we'll do our own robust-loss polish below
#         updating="deferred",
#         # workers=-1,     # parallelize across cores; drop this if it causes issues on your setup
#     )
#     print(f"  DE best cost (L2, normalized) = {de_result.fun:.6f}")

#     # ---- polish with robust loss to guard against outliers ----
#     print("stage 2b — polishing with robust loss (least_squares, soft_l1)...")
#     polish_result = least_squares(
#         residuals,
#         de_result.x,
#         bounds=(lb, ub),
#         method="trf",
#         loss="cauchy",
#         f_scale=0.05,   # tune against your residual scale — see note below
#     )

#     p_fit = expand(polish_result.x)
#     rms_err = np.sqrt(np.mean((FY - _mf_lateral(p_fit, SA, FZ, IA)) ** 2))
#     print(f"fit complete — rms error: {rms_err:.2f} N\n")

#     coeff_names = ["a0 (C)", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
#                    "a8", "a9", "a10", "a11", "a12", "a13"]
#     print("fitted coefficients:")
#     for name, val in zip(coeff_names, p_fit):
#         print(f"  {name:<14} = {val:12.6f}")

#     savemat(Path("../../data/coeffs/hoosier_r20_tire_params_lat.mat").expanduser().resolve(), {
#         "coeffs": p_fit,
#         "coeff_names": coeff_names,
#         "rms_error_N": rms_err,
#         "tire": "Hoosier 43075 16x7.5-10 R20",
#         "source": "FSAE TTC Round 9, Calspan Tire Research Facility",
#         "model": "Pacejka BNP 1989 Lateral",
#     })
#     print("tire parameters saved to hoosier_r20_tire_params_lat.mat")

#     _coeffs_cache = p_fit
#     return p_fit

def generate_coeffs(n_starts=8, seed=0):
    """
    Two-stage fit:
      stage 1 — anchor D(Fz) = a1*Fz^2 + a2*Fz to the data's peak |FY| per load bin
      stage 2 — multi-start bounded nonlinear least squares on the remaining
                 12 shape parameters, with a1/a2 held fixed at stage-1 values
    Returns the fitted 14-element coefficient array and caches it.
    """
    global _coeffs_cache
    if _coeffs_cache is not None:
        return _coeffs_cache

    data = load_data()
    SA, FY, FZ, IA = data["SA"], data["FY"], data["FZ"], data["IA"]

    # ---- stage 1: anchor peak-force load curve ----
    fz_bin_centers = np.arange(150, 1251, 50)
    fz_bin_tol = 55.0
    fzc, dpk = [], []
    for fz in fz_bin_centers:
        sel = (np.abs(FZ - fz) < fz_bin_tol) & (np.abs(SA) > 5) & (np.abs(IA) < 0.6)
        if sel.sum() > 40:
            v = np.sort(np.abs(FY[sel]))
            fzc.append(fz)
            dpk.append(v[max(0, round(0.90 * len(v)) - 1)])  # 90th percentile proxy
    fzc = np.asarray(fzc, dtype=float)
    dpk = np.asarray(dpk, dtype=float)

    A = np.column_stack([fzc**2, fzc])
    ab, *_ = np.linalg.lstsq(A, -dpk, rcond=None)
    a1_fix, a2_fix = ab
    print(f"stage 1 — anchored load curve: a1 = {a1_fix:.4e}, a2 = {a2_fix:.4f}")
    print(f"          peak mu  {abs(a1_fix*200 + a2_fix):.2f} (at 200 N) -> "
          f"{abs(a1_fix*1200 + a2_fix):.2f} (at 1200 N)")

    # ---- stage 2: multi-start fit of remaining 12 shape parameters ----
    # free vector order: [a0, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13]
    def expand(free):
        a0, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 = free
        return np.array([a0, a1_fix, a2_fix, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13])

    def residuals(free):
        p = expand(free)
        model = _mf_lateral(p, SA, FZ, IA)
        return (FY - model) / FZ  # load-normalized residual

    lb = np.array([1.2, -5000, 100, 0.0, -5e-4, -1.0, -2, -1e-2, -5, -10, -2, -200])
    ub = np.array([1.8, 0, 5000, 0.5, 5e-4, 1.5, 2, 1e-2, 5, 10, 2, 200])

    p_nom_free = np.array([1.45, -800, 1900, 0.01, 0.0, 0.60, 0.0, 1e-5, 0.0, 0.0, 0.0, 0.0])

    rng = np.random.default_rng(seed)
    best_cost = np.inf
    best_free = p_nom_free

    print(f"stage 2 — fitting shape parameters ({n_starts} starts)...")
    for s in range(n_starts):
        x0 = p_nom_free if s == 0 else rng.uniform(lb, ub)
        try:
            result = least_squares(residuals, x0, bounds=(lb, ub), method="trf", loss="cauchy", f_scale=0.05)
        except Exception as err:
            print(f"  start {s+1}/{n_starts}  skipped ({err})")
            continue

        cost = 2 * result.cost  # least_squares cost is 0.5*sum(res^2)
        print(f"  start {s+1}/{n_starts}  normalized cost = {cost:.5f}")
        if cost < best_cost:
            best_cost = cost
            best_free = result.x

    p_fit = expand(best_free)
    rms_err = np.sqrt(np.mean((FY - _mf_lateral(p_fit, SA, FZ, IA)) ** 2))
    print(f"\nbest normalized cost = {best_cost:.5f}")
    print(f"fit complete — rms error: {rms_err:.2f} N\n")

    coeff_names = ["a0 (C)", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
                   "a8", "a9", "a10", "a11", "a12", "a13"]
    print("fitted coefficients:")
    for name, val in zip(coeff_names, p_fit):
        print(f"  {name:<14} = {val:12.6f}")

    savemat(Path("../../data/coeffs/hoosier_r20_tire_params_lat.mat").expanduser().resolve(), {
        "coeffs": p_fit,
        "coeff_names": coeff_names,
        "rms_error_N": rms_err,
        "tire": "Hoosier 43075 16x7.5-10 R20",
        "source": "FSAE TTC Round 9, Calspan Tire Research Facility",
        "model": "Pacejka BNP 1989 Lateral",
    })
    print("\ntire parameters saved to hoosier_r20_tire_params.mat")

    _coeffs_cache = p_fit
    return p_fit


def plot_fy_vs_sa(fn_buckets):
    """
    Lateral force vs slip angle, one curve per Fz bucket in fn_buckets [N].
    Model curves (zero camber) overlaid on measured scatter near each bucket.
    """
    data = load_data()
    SA, FY, FZ, IA = data["SA"], data["FY"], data["FZ"], data["IA"]
    p_fit = generate_coeffs()

    sa_vec = np.linspace(-14, 14, 300)
    fz_tol = 60.0
    cmap = plt.get_cmap("jet", len(fn_buckets))

    fig, ax = plt.subplots(figsize=(9, 6))
    for k, fz in enumerate(fn_buckets):
        idx = (np.abs(FZ - fz) < fz_tol) & (np.abs(IA) < 0.5)
        if idx.sum() < 30:
            print(f"skipping F_Z = {fz} N (only {idx.sum()} points nearby)")
            continue

        ax.scatter(SA[idx], FY[idx], s=5, color=cmap(k), alpha=0.20)

        fy_pred = _mf_lateral(p_fit, sa_vec, fz * np.ones_like(sa_vec), 0.0)
        ax.plot(sa_vec, fy_pred, color=cmap(k), linewidth=2, label=f"F_Z = {fz:g} N")

    ax.axhline(0, linestyle="--", color="k", alpha=0.25)
    ax.axvline(0, linestyle="--", color="k", alpha=0.25)
    ax.set_xlabel("slip angle  α  [deg]")
    ax.set_ylabel("lateral force  F_Y  [N]")
    ax.set_title("lateral force vs slip angle (lines = model, dots = measured)")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(Path(f"../../figures/tyres/Fy vs SA.png").expanduser().resolve())


def plot_fy_vs_fn(sa_buckets):
    """
    Lateral force vs normal force, one curve per slip angle in sa_buckets [deg].
    Model-only (continuous Fz sweep, zero camber) — this is the direct
    visualization of load sensitivity: each curve's sub-linear rise with Fz
    shows the diminishing-returns behavior discussed earlier.
    """
    p_fit = generate_coeffs()

    fz_vec = np.linspace(100, 1300, 300)
    cmap = plt.get_cmap("viridis", len(sa_buckets))

    fig, ax = plt.subplots(figsize=(9, 6))
    for k, sa in enumerate(sa_buckets):
        fy_pred = np.abs(_mf_lateral(p_fit, sa * np.ones_like(fz_vec), fz_vec, 0.0))
        ax.plot(fz_vec, fy_pred, color=cmap(k), linewidth=2, label=f"α = {sa:g}°")

    ax.set_xlabel("normal force  F_Z  [N]")
    ax.set_ylabel("lateral force  |F_Y|  [N]")
    ax.set_title("lateral force vs normal force — load sensitivity")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(Path(f"../../figures/tyres/Fy vs FN.png").expanduser().resolve())



if __name__ == "__main__":
    load_data()
    generate_coeffs()
    plot_fy_vs_sa(fn_buckets=[200, 600, 1000, 1200])
    plot_fy_vs_fn(sa_buckets=[1.92, 4, 8, 12])