"""Plot helpers for cached MicTherm grids (consistency check of critical point, VLE and p grid).

Run directly to check cached grids: py -3.11 MicLBM_src/grid_plot.py [fluid ...]
(default: VdW_SI_reduced CO2 PeTS; fluid = suffix of temp/temp_mictherm_grids_<fluid>.npz, e.g. VdW for the SI cache).
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MicLBM_src.eos import interpolate_mictherm_properties

GRID_KEYS = ("rho_grid", "p_grid", "T_grid", "eta_grid", "gamma_surface_grid")


def plot_vle_on_pressure_grid(ax, grid, verbose=True):
    """Add the VLE curve and the critical point to a 3D p(rho, T) surface plot on ``ax``.

    ``grid`` holds the cache keys (Tc, rhoc, pc, VLE_T, VLE_rho_l, VLE_rho_g and the property grids)
    in the units of the cache. p on the VLE curve is interpolated from p_grid; T is limited to the grid
    range (no extrapolation). Consistent units: liquid and gas branch have equal p, and the critical
    point continues the surface. Returns max |p_l - p_g| / pc (interpolation error included).
    """
    T_min, T_max = np.min(grid["T_grid"]), np.max(grid["T_grid"])
    VLE_T = np.asarray(grid["VLE_T"])
    in_grid = (VLE_T >= T_min) & (VLE_T <= T_max)
    VLE_T = VLE_T[in_grid]
    VLE_rho_l = np.asarray(grid["VLE_rho_l"])[in_grid]
    VLE_rho_g = np.asarray(grid["VLE_rho_g"])[in_grid]
    grid_kwargs = {key: grid[key] for key in GRID_KEYS}
    VLE_p_l = np.asarray(interpolate_mictherm_properties(kwargs=grid_kwargs, rho=VLE_rho_l, T=VLE_T)["p"])
    VLE_p_g = np.asarray(interpolate_mictherm_properties(kwargs=grid_kwargs, rho=VLE_rho_g, T=VLE_T)["p"])
    mismatch = float(np.max(np.abs(VLE_p_l - VLE_p_g)) / grid["pc"]) if VLE_T.size else float("nan")
    if verbose:
        print(f"Critical point: Tc={grid['Tc']:.4f}, rhoc={grid['rhoc']:.4f}, pc={grid['pc']:.4f}")
        print(f"VLE in grid range: {VLE_T.size} points, max |p_l - p_g| / pc = {mismatch:.3e}")

    ax.plot(VLE_rho_l, VLE_T, VLE_p_l, color="blue", lw=2, label=r"VLE $\rho_l$")
    ax.plot(VLE_rho_g, VLE_T, VLE_p_g, color="red", lw=2, label=r"VLE $\rho_g$")
    ax.scatter([grid["rhoc"]], [grid["Tc"]], [grid["pc"]], color="black", s=40, depthshade=False, label="critical point")
    ax.legend(loc="upper left", fontsize=8)
    return mismatch


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # Default: reduced caches only (VdW SI_reduced, CO2 SI_reduced, PeTS reduced); "VdW" selects the SI cache.
    fluids = sys.argv[1:] or ["VdW_SI_reduced", "CO2", "PeTS"]
    fig = plt.figure(figsize=(6 * len(fluids), 5))
    for i, fluid in enumerate(fluids):
        with np.load(PROJECT_ROOT / "temp" / f"temp_mictherm_grids_{fluid}.npz") as data:
            grid = {key: (data[key].item() if data[key].ndim == 0 else data[key]) for key in data.files}
        print(f"--- {fluid}: grid {grid['T_grid'].shape}")
        ax = fig.add_subplot(1, len(fluids), i + 1, projection="3d")
        ax.plot_surface(grid["rho_grid"], grid["T_grid"], grid["p_grid"], cmap=cm.nipy_spectral, alpha=0.5)
        plot_vle_on_pressure_grid(ax, grid)
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$T$")
        ax.set_zlabel(r"$p$")
        ax.set_title(f"{fluid} (cache units)")
    fig.tight_layout()
    plt.show()
