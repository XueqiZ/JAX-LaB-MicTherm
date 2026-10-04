"""
Single component 2D droplet example where liquid droplet is suspended in its vapor. The density of each region is computed using Maxwell's Construction. The density profile
is initialized with smooth profile with specified interface width. Boundary conditions are periodic everywhere. Useful for tuning the various coefficients.

The collision matrix is based on:
1. McCracken, M. E. & Abraham, J. Multiple-relaxation-time lattice-Boltzmann model for multiphase flow. Phys. Rev. E 71, 036701 (2005).

The simulation setup lives in load_grid / generate_grid / build_simulation so that tests/smoke_vdw_droplet.py can reuse it.
"""

import os
import sys
from pathlib import Path
from jax import config
import numpy as np
from scipy.interpolate import interp1d

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MicLBM_src import *
from MicLBM_src.run_mictherm import (
    extract_mictherm_properties,
    mictherm_grid,
    run_mictherm_func,
)
from src.lattice import LatticeD2Q9
from MicLBM_src.eos import MicTherm, interpolate_mictherm_properties
from MicLBM_src.grid_plot import plot_vle_on_pressure_grid
from src.utils import *
from MicLBM_src.Mic_multiphase import MultiphaseMRTTvar


# config.update("jax_default_matmul_precision", "float32")

# One cache per unit system, so SI and reduced grids are never mixed up (load_grid checks the stored "units").
GRID_FILES = {
    "SI": PROJECT_ROOT / "temp" / "temp_mictherm_grids_VdW.npz",
    "SI_reduced": PROJECT_ROOT / "temp" / "temp_mictherm_grids_VdW_SI_reduced.npz",
}
GRID_FILE = GRID_FILES["SI"]
# Axis labels (rho, T, p) per unit system.
AXIS_LABELS = {
    "SI": (r"$\rho$ / mol L$^{-1}$", r"$T$ / K", r"$p$ / MPa"),
    "SI_reduced": (r"$\rho^*$", r"$T^*$", r"$p^*$"),
}


class Droplet2D(MultiphaseMRTTvar):
    def __init__(self, **kwargs):
        # Droplet setup; extra kwargs are ignored by the base classes.
        self.rho_l = kwargs["rho_l"]
        self.rho_g = kwargs["rho_g"]
        self.radius = kwargs.get("radius", 30)
        self.width = kwargs.get("width", 3)
        self.output_dir = kwargs.get("output_dir", "output")
        self.last_metrics = None
        super().__init__(**kwargs)

    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)

        rho_tree = []

        dist = np.sqrt((x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2)

        rho = 0.5 * (self.rho_l + self.rho_g) - 0.5 * (self.rho_l - self.rho_g) * np.tanh(2 * (dist - self.radius) / self.width)

        rho = rho.reshape((self.nx, self.ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u]
        return rho_tree, u_tree

    def output_data(self, **kwargs):
        rho_l, rho_g = self.rho_l, self.rho_g
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        p = np.array(kwargs["p"][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        T_field = np.array(self.T_field)
        fields = {"p": p[..., 0], "rho": rho[..., 0], "T": T_field[..., 0], "ux": u[..., 0], "uy": u[..., 1]}
        offset = 90
        rho_north = rho[self.nx // 2, self.ny // 2 - offset, 0]
        rho_south = rho[self.nx // 2, self.ny // 2 + offset, 0]
        rho_west = rho[self.nx // 2 - offset, self.ny // 2, 0]
        rho_east = rho[self.nx // 2 + offset, self.ny // 2, 0]
        rho_g_pred = 0.25 * (rho_north + rho_south + rho_west + rho_east)
        rho_l_pred = rho[self.nx // 2, self.ny // 2, 0]
        spurious_currents = np.max(np.sqrt(np.sum(u**2, axis=-1)))
        print(f"%Error Min: {(rho_g_pred - rho_g) * 100 / rho_g} Max: {(rho_l_pred - rho_l) * 100 / rho_l}")
        print(f"Density: Min: {rho_g_pred} Max: {rho_l_pred}")
        print(f"Maxwell construction: Min: {rho_g} Max: {rho_l}")
        print(f"Spurious currents: {spurious_currents}")
        p_north = p[self.nx // 2, self.ny // 2 - offset, 0]
        p_south = p[self.nx // 2, self.ny // 2 + offset, 0]
        p_west = p[self.nx // 2 - offset, self.ny // 2, 0]
        p_east = p[self.nx // 2 + offset, self.ny // 2, 0]
        pressure_difference = p[self.nx // 2, self.ny // 2, 0] - 0.25 * (p_north + p_south + p_west + p_east)
        print(f"Pressure difference: {pressure_difference}")

        self.last_metrics = {
            "timestep": timestep,
            "all_finite": bool(np.all(np.isfinite(rho)) and np.all(np.isfinite(p)) and np.all(np.isfinite(u))),
            "rho_l_pred": float(rho_l_pred),
            "rho_g_pred": float(rho_g_pred),
            "error_rho_l_percent": float((rho_l_pred - rho_l) * 100 / rho_l),
            "error_rho_g_percent": float((rho_g_pred - rho_g) * 100 / rho_g),
            "spurious_currents": float(spurious_currents),
            "pressure_difference": float(pressure_difference),
        }

        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        x_mid = self.nx // 2
        y_positions = np.arange(self.ny)
        pressure_profile = p[x_mid, :, 0]
        pressure_profile_data = np.column_stack((y_positions, pressure_profile))
        np.savetxt(
            os.path.join(output_dir, f"pressure_profile_x_mid_{str(timestep).zfill(7)}.csv"),
            pressure_profile_data,
            delimiter=",",
            header=f"y,pressure_at_x_{x_mid}",
            comments="",
        )

        save_fields_vtk(timestep, fields, output_dir, "data")
        save_image(timestep, u, prefix=os.path.join(output_dir, ""))


MIC_THERM_USER_PARAMETERS = {
        # General
        "stability": "no",
        "N_components": 1,
        "Substance_ID1": 0,
        "PotModel_1": "LJ.pm",
        "units": "SI",
        "EOS": "vanderWaals",
        "Output": "no",
        "Debug": "no",
        "chainlength_1": 1,
        # Lattice VdW a = 9/49, b = 2/21 scaled by 1000 (MicTherm's units for a, b). Then rho [mol/L] and
        # p [MPa] equal the lattice VdW values (rho_c = 3.5, p_c = 0.75); T [K] = T_lattice * 1000 / R
        # (T_c = 68.73 K instead of 4/7). T is only the lookup coordinate of p(rho, T), so the LBM is unchanged.
        #
        # Why T_c != 4/7 (intended, not a bug):
        #   VdW: p = rho*R*T/(1 - b*rho) - a*rho^2  ->  rho_c = 1/(3b), p_c = a/(27 b^2), T_c = 8a/(27 b R)
        #   Only T_c contains R. Lattice: R = 1 -> T_c = 4/7. MicTherm (mol/L, MPa): R_eff = R/1000
        #   = 0.008314 MPa L/(mol K) -> T_c = (4/7)/0.008314 = 68.73 K, while rho_c and p_c stay 3.5 and 0.75.
        #   Since Z_c = p_c/(rho_c R T_c) = 3/8 for every a, b, at most two of the three lattice values
        #   (rho_c, p_c, T_c) can be matched with R_eff != 1. We match rho and p (used directly by the LBM);
        #   T always enters as Tr = T/T_c, so the T field is the lattice case with the axis scaled by 1000/R.
        #   Checked 2026-10-04: p_grid equals the lattice VdW p(rho, T*R/1000) to 1e-9 * p_c.
        "b_VDW_1": 1000 * 2 / 21,
        "a_VDW_1": 1000 * 9 / 49,
        "molar_mass_1": 114.04,
        "eta_param": "0,-16.887065518,15.000907502,-2.8455190264,0.44056790447", #data from Argon, didn't have any physical meaning, just to get a reasonable viscosity value for the simulation
        "DGT_kappa_1": 6.738872, #ditto
        "transPropMode": "entropyScaling",
        "properties": "p, eta, gamma_surface",
}


def load_grid(temp_file=None, units="SI"):
    """Load the cached MicTherm VdW grid and VLE data in the given unit system (see generate_grid).

    Caches without a "units" key were written before the key existed and are SI.
    """
    temp_file = Path(temp_file or GRID_FILES[units])
    if not temp_file.exists():
        raise FileNotFoundError(
            f"{temp_file} does not exist. Run once with grid mode 2 "
            "to calculate and cache the MicTherm data."
        )
    with np.load(temp_file) as data:
        cached_units = str(data["units"]) if "units" in data.files else "SI"
        if cached_units != units:
            raise ValueError(f"{temp_file} holds a {cached_units} grid, but units = {units} was requested.")
        return {
            "units": cached_units,
            "Tc": data["Tc"].item(),
            "rhoc": data["rhoc"].item(),
            "pc": data["pc"].item(),
            "VLE_T": data["VLE_T"],
            "VLE_rho_l": data["VLE_rho_l"],
            "VLE_rho_g": data["VLE_rho_g"],
            "T_grid": data["T_grid"],
            "rho_grid": data["rho_grid"],
            "p_grid": data["p_grid"],
            "eta_grid": data["eta_grid"],
            "gamma_surface_grid": data["gamma_surface_grid"],
        }


def vle_interpolators(grid):
    """Return interpolators T -> liquid/gas density on the VLE curve (units of the grid)."""
    interp_rho_l = interp1d(grid["VLE_T"], grid["VLE_rho_l"], kind="cubic", fill_value="extrapolate")
    interp_rho_g = interp1d(grid["VLE_T"], grid["VLE_rho_g"], kind="cubic", fill_value="extrapolate")
    return interp_rho_l, interp_rho_g


def generate_grid(temp_file=None, units="SI", Trmin=0.3, Tstep=0.01, rhostep=0.03):
    """Compute the VdW grid with MicTherm (runtime required) and save it to temp_file.

    units = SI: critical point, VLE and grid are calculated and saved in K, mol/L, MPa (eta in Pa s,
    gamma_surface in N/m) without any conversion. See MIC_THERM_USER_PARAMETERS for why rho and p
    equal the lattice VdW values.

    units = SI_reduced (same pattern as the CO2 script): MicTherm returns reduced values (critical point,
    VLE, p, eta, gamma_surface), but expects the state inputs T and rho of "userproperties" in SI (K, mol/L).
    The grid is built in reduced units and converted to SI only for that call; everything returned and
    saved is reduced. The reduction is a fixed scaling (eps/k = 200 K, sigma = 3.5 A; tested 2026-10-04:
    T* = T / 200 K, rho* = rho / 38.73, p* = p / 64.40); epsilon_1 / sigma_1 have no effect for vanderWaals.
    """
    temp_file = temp_file or GRID_FILES[units]
    params = {**MIC_THERM_USER_PARAMETERS, "units": units}

    # Critical point (in the chosen units).
    mictherm_names, mictherm_units, mictherm_values = run_mictherm_func(
        mode="criticalpoint", T=None, rho=None, p=None, x=None, t_iso=None,
        init_mode="uninitialized", print_output=False,
        base_user_parameters=params,
    )
    Tc, rhoc, pc = mictherm_values[0, 0], mictherm_values[0, 1], mictherm_values[0, 2]

    # Conversion factors SI / chosen units, only for the "userproperties" input (1 for SI).
    factorTc, factorRho = 1.0, 1.0
    if units == "SI_reduced":
        mictherm_names, mictherm_units, mictherm_values = run_mictherm_func(
            mode="criticalpoint", T=None, rho=None, p=None, x=None, t_iso=None,
            init_mode="uninitialized", print_output=False,
            base_user_parameters={**params, "units": "SI"},
        )
        factorTc = mictherm_values[0, 0] / Tc
        factorRho = mictherm_values[0, 1] / rhoc

    # VLE curve (in the chosen units).
    mictherm_names, mictherm_units, mictherm_values = run_mictherm_func(
        mode="vle_full", T=None, rho=None, p=None, x=1,
        init_mode="uninitialized", print_output=False,
        base_user_parameters=params,
    )
    micthermVLE_T = mictherm_values[:, 0]
    micthermVLE_rho_l = mictherm_values[:, 1]
    micthermVLE_rho_g = mictherm_values[:, 2]

    interp_rho_l = interp1d(micthermVLE_T, micthermVLE_rho_l, kind="cubic", fill_value="extrapolate")
    rho_l_max = interp_rho_l(Trmin * Tc)

    # State grid (in the chosen units); MicTherm gets it in SI (K, mol/L).
    T_grid = np.arange(Trmin, 1.0, Tstep) * Tc
    rho_grid = np.arange(0.01, rho_l_max / rhoc, rhostep) * rhoc
    rho_grid_mesh, T_grid_mesh = np.meshgrid(rho_grid, T_grid)
    T_array = T_grid_mesh.flatten()
    rho_array = rho_grid_mesh.flatten()

    mictherm_names, mictherm_units, mictherm_values, InputT, Inputp, Inputrho, Inputx = mictherm_grid(
        mode="userproperties",
        T=T_array * factorTc,
        rho=rho_array * factorRho,
        p=None,
        x=np.ones_like(T_array),
        print_output=False,
        base_user_parameters=params,
    )
    mictherm_properties = extract_mictherm_properties(
        MIC_THERM_USER_PARAMETERS["properties"],
        mictherm_values,
    )
    grid_shape = (T_grid.shape[0], rho_grid.shape[0])
    grid = {
        "units": units,
        "Tc": Tc,
        "rhoc": rhoc,
        "pc": pc,
        "VLE_T": micthermVLE_T,
        "VLE_rho_l": micthermVLE_rho_l,
        "VLE_rho_g": micthermVLE_rho_g,
        "T_grid": T_array.reshape(grid_shape),
        "rho_grid": rho_array.reshape(grid_shape),
        "p_grid": mictherm_properties["p"].reshape(grid_shape),
        "eta_grid": mictherm_properties["eta"].reshape(grid_shape),
        "gamma_surface_grid": mictherm_properties["gamma_surface"].reshape(grid_shape),
    }
    Path(temp_file).parent.mkdir(parents=True, exist_ok=True)
    np.savez(temp_file, **grid)
    return grid


def plot_grid(grid):
    fig = plt.figure(figsize=(16, 5))

    # Subplot 1: Pressure grid with VLE curve and critical point (unit consistency check).
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(grid["rho_grid"], grid["T_grid"], grid["p_grid"], cmap=cm.nipy_spectral, alpha=0.5)
    plot_vle_on_pressure_grid(ax1, grid)
    rho_label, T_label, p_label = AXIS_LABELS[grid["units"]]
    ax1.set_xlabel(rho_label)
    ax1.set_ylabel(T_label)
    ax1.set_zlabel(p_label)
    ax1.set_title("MicTherm pressure grid (3D)")

    # Subplot 2: Viscosity grid
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(grid["rho_grid"], grid["T_grid"], grid["eta_grid"], cmap=cm.viridis, alpha=0.8)
    ax2.set_xlabel(r"$\rho$")
    ax2.set_ylabel(r"$T$")
    ax2.set_zlabel(r"$\eta$")
    ax2.set_title("MicTherm viscosity grid (3D)")

    # Subplot 3: Surface tension grid
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(grid["rho_grid"], grid["T_grid"], grid["gamma_surface_grid"], cmap=cm.plasma, alpha=0.8)
    ax3.set_xlabel(r"$\rho$")
    ax3.set_ylabel(r"$T$")
    ax3.set_zlabel(r"$\gamma_{surface}$")
    ax3.set_title("MicTherm surface tension grid (3D)")

    plt.tight_layout()
    plt.show()


def build_simulation(grid, Tr=0.8, output_dir="output", nx=200, ny=200, radius=30, width=3, io_rate=10, print_info_rate=10):
    """Set up the VdW droplet with a vertical temperature gradient (T, rho, p in the units of the grid)."""
    if grid["units"] != "SI":
        # k and A below are tuned for the SI grid (= lattice VdW, rho_c = 3.5). With SI_reduced, rho_c* = 0.09
        # and p_c* = 0.0116, so k and A (and possibly g) are untested and likely need retuning.
        print(f"WARNING: k, A are tuned for the SI grid; untested with units = {grid['units']}.")
    T = Tr * grid["Tc"]
    interp_rho_l, interp_rho_g = vle_interpolators(grid)
    rho_l = interp_rho_l(T)
    rho_g = interp_rho_g(T)

    e = LatticeD2Q9().c.T
    en = np.linalg.norm(e, axis=1)

    M = np.zeros((9, 9))
    M[0, :] = en**0
    M[1, :] = -4 * en**0 + 3 * en**2
    M[2, :] = 4 * en**0 - (21 / 2) * en**2 + (9 / 2) * en**4
    M[3, :] = e[:, 0]
    M[4, :] = (-5 * en**0 + 3 * en**2) * e[:, 0]
    M[5, :] = e[:, 1]
    M[6, :] = (-5 * en**0 + 3 * en**2) * e[:, 1]
    M[7, :] = e[:, 0] ** 2 - e[:, 1] ** 2
    M[8, :] = e[:, 0] * e[:, 1]

    T_l = 0.9 * T
    T_g = 1.1 * T
    # Vertical temperature profile: top (y=0) is hotter (T_g), bottom (y=ny-1) is colder (T_l)
    # normalized vertical coordinate (0 at top, 1 at bottom)
    y_norm = np.linspace(0.0, 1.0, ny, dtype=float)
    T_field = np.tile(T_l + (T_g - T_l) * y_norm, (nx, 1))
    T_field = T_field.reshape((nx, ny, 1))

    # T_field = 0.5 * (T_l + T_g) - 0.5 * (T_l - T_g) * np.tanh(2 * (dist - r) / width)
    # T_field = T_field.reshape((nx, ny, 1))

    s_rho = [0.0]
    s_e = [1.2]
    s_eta = [1.0]
    s_j = [0.0]
    s_q = [1.0]
    s_v = [1.0]

    # Note for Martin: I don't know on which position you want to use eta_grid and gamma_surface_grid, so I just saved them in kwargs for now. You can access them later in the code as needed.
    # for rho,p,T, the grid intepolation has been carried out in MicTherm class, so you can access them directly from the eos object.
    # Units: SI -> rho mol/L, p MPa, T K, eta Pa s, gamma_surface N/m; SI_reduced -> all reduced.
    eos_kwargs = {
        "rho_grid": grid["rho_grid"],
        "p_grid": grid["p_grid"], # the p_grid has been saved here in kwargs for later use in the further code
        "T_grid": grid["T_grid"],
        "eta_grid": grid["eta_grid"],  # the eta_grid has been saved here in kwargs for later use in the further code
        "gamma_surface_grid": grid["gamma_surface_grid"], # the gamma_surface has been saved here in kwargs for later use in the further code
    }
    eos = MicTherm(**eos_kwargs)

    precision = "f32/f32"
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD2Q9(precision),
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "g_kkprime": -1.0 * np.ones((1, 1)),
        "EOS": eos,
        "T_field": T_field,
        "body_force": [0.0, 0.0],
        "k": [0.02416],
        "A": -0.258127 * np.ones((1, 1)),
        "M": [M],
        "s_rho": s_rho,
        "s_e": s_e,
        "s_eta": s_eta,
        "s_j": s_j,
        "s_q": s_q,
        "s_v": s_v,
        "kappa": [0.9],
        "precision": precision,
        "io_rate": io_rate,
        "compute_MLUPS": False,
        "print_info_rate": print_info_rate,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
        # Droplet2D setup
        "rho_l": rho_l,
        "rho_g": rho_g,
        "radius": radius,
        "width": width,
        "output_dir": output_dir,
    }
    return Droplet2D(**kwargs)


if __name__ == "__main__":
    def _inp(prompt, default):
        v = input(f"{prompt} [{default}]: ")
        return v.strip() or str(default)

    print("Select grid mode:\n 1) Use existing precomputed grid (faster)\n 2) Generate new grid (full recompute)")
    mode = _inp("Choose 1 or 2", "1")
    debugging = 1 if mode == "1" else 0
    print("Select MicTherm units:\n 1) SI (same as the original lattice test case: rho, p = lattice VdW values)\n"
          " 2) SI_reduced (reduced outputs; k, A untested)")
    units = "SI_reduced" if _inp("Choose 1 or 2", "1") == "2" else "SI"
    Tr = float(_inp("Reduced temperature Tr", 0.8))

    Trmin = 0.3
    Tstep = 0.01
    rhostep = 0.03
    print(f"Current grid defaults: Trmin={Trmin}, Tstep={Tstep}, rhostep={rhostep}.")
    print("If these are not suitable, regenerate the grid (mode 2); MicTherm runtime is required.")
    if debugging == 0:
        Trmin = float(_inp("Trmin (min reduced temperature)", Trmin))
        Tstep = float(_inp("Tstep (temperature step)", Tstep))
        rhostep = float(_inp("rhostep (density step)", rhostep))
        grid = generate_grid(units=units, Trmin=Trmin, Tstep=Tstep, rhostep=rhostep)
    else:
        print(f"Using precomputed grids. Current Trmin={Trmin}, Tstep={Tstep}, rhostep={rhostep}.")
        print("To change these values, choose mode 2 (Generate new grid), because regeneration requires MicTherm runtime.")
        grid = load_grid(units=units)

    plot_grid(grid)

    # example of how to interpolate properties at a specific state point (units of the grid)
    properties = interpolate_mictherm_properties(
        kwargs={key: grid[key] for key in ("rho_grid", "p_grid", "T_grid", "eta_grid", "gamma_surface_grid")},
        rho=0.85 * grid["rhoc"],
        T=0.8 * grid["Tc"],
        )

    sim = build_simulation(grid, Tr=Tr, output_dir="output")
    sim.run(200)
