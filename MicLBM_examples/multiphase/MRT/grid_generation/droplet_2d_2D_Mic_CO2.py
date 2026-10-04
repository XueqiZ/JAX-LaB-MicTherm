"""
Single component 2D droplet example where liquid droplet is suspended in its vapor. The density of each region is computed using Maxwell's Construction. The density profile
is initialized with smooth profile with specified interface width. Boundary conditions are periodic everywhere. Useful for tuning the various coefficients.

The collision matrix is based on:
1. McCracken, M. E. & Abraham, J. Multiple-relaxation-time lattice-Boltzmann model for multiphase flow. Phys. Rev. E 71, 036701 (2005).
"""

import os
import sys
from pathlib import Path
from jax import config
import numpy as np
from pathlib import Path
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


class Droplet2D(MultiphaseMRTTvar):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)

        rho_tree = []

        dist = np.sqrt((x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2)

        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (dist - r) / width)

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
        print(f"%Error Min: {(rho_g_pred - rho_g) * 100 / rho_g} Max: {(rho_l_pred - rho_l) * 100 / rho_l}")
        print(f"Density: Min: {rho_g_pred} Max: {rho_l_pred}")
        print(f"Maxwell construction: Min: {rho_g} Max: {rho_l}")
        print(f"Spurious currents: {np.max(np.sqrt(np.sum(u**2, axis=-1)))}")
        p_north = p[self.nx // 2, self.ny // 2 - offset, 0]
        p_south = p[self.nx // 2, self.ny // 2 + offset, 0]
        p_west = p[self.nx // 2 - offset, self.ny // 2, 0]
        p_east = p[self.nx // 2 + offset, self.ny // 2, 0]
        pressure_difference = p[self.nx // 2, self.ny // 2, 0] - 0.25 * (p_north + p_south + p_west + p_east)
        print(f"Pressure difference: {pressure_difference}")

        output_dir = "output"
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

        save_fields_vtk(timestep, fields, "output", "data")
        save_image(timestep, u)
        
MIC_THERM_USER_PARAMETERS = {
        # General
        "units": "SI_reduced",
        "Output": "no",
        "Debug": "no",
        "stability": "no",

        # EOS / model settings
        "N_components": 1,
        "Substance_ID1": 0,
        "PotModel_1": "LJ.pm",
        "EOS": "M_SAFT_VR_MIE",
        "IDEAL": "IdealQM",

        # Substance parameters
        "epsilon_1": 182.32,
        "sigma_1": 3.7131,
        "chainlength": 1.0677,
        "molar_mass_1": 44.009,
        "CAS_number_1": "124-38-9",
        "Quadrupolemoment_1": 4.4,
        "N_Quadrupolemoment_1": 1,
        "eta_param": "0,-2.5399,2.9133,-0.2182,0.1340",
        "DGT_kappa_1": 2.5726,
        "transPropMode": "entropyScaling",
        "properties": "p, eta, gamma_surface",
}


def generate_grid(temp_file, Trmin=0.3, Tstep=0.01, rhostep=0.03):
    """Compute the CO2 grid with MicTherm (runtime required) and save it to temp_file.

    With units = SI_reduced, MicTherm returns reduced values (critical point, VLE, p, eta, gamma_surface),
    but expects the state inputs T and rho of "userproperties" in SI (K, mol/L).
    The grid is built in reduced units and converted to SI only for the MicTherm call.
    All returned and saved grids are in reduced units.
    """
    # Critical point in reduced units.
    mictherm_names, mictherm_units, mictherm_values = run_mictherm_func(
        mode="criticalpoint", T=None, rho=None, p=None, x=None, t_iso=None,
        init_mode="uninitialized", print_output=False,
        base_user_parameters=MIC_THERM_USER_PARAMETERS,
    )
    Tc, rhoc, pc = mictherm_values[0, 0], mictherm_values[0, 1], mictherm_values[0, 2]

    # Critical point in SI (K, mol/L, MPa) -> conversion factors SI / reduced (only needed for the MicTherm input).
    mictherm_names, mictherm_units, mictherm_values = run_mictherm_func(
        mode="criticalpoint", T=None, rho=None, p=None, x=None, t_iso=None,
        init_mode="uninitialized", print_output=False,
        base_user_parameters={**MIC_THERM_USER_PARAMETERS, "units": "SI"},
    )
    factorTc = mictherm_values[0, 0] / Tc
    factorRho = mictherm_values[0, 1] / rhoc

    # VLE curve in reduced units.
    mictherm_names, mictherm_units, mictherm_values = run_mictherm_func(
        mode="vle_full", T=None, rho=None, p=None, x=1,
        init_mode="uninitialized", print_output=False,
        base_user_parameters=MIC_THERM_USER_PARAMETERS,
    )
    micthermVLE_T = mictherm_values[:, 0]
    micthermVLE_rho_l = mictherm_values[:, 1]
    micthermVLE_rho_g = mictherm_values[:, 2]

    interp_rho_l = interp1d(micthermVLE_T, micthermVLE_rho_l, kind="cubic", fill_value="extrapolate")
    rho_l_max = interp_rho_l(Trmin * Tc)

    # Grid in reduced units; MicTherm gets it in SI (K, mol/L).
    reduced_T_grid = np.arange(Trmin, 1.0, Tstep) * Tc
    reduced_rho_grid = np.arange(0.01, rho_l_max / rhoc, rhostep) * rhoc
    rho_grid_mesh, T_grid_mesh = np.meshgrid(reduced_rho_grid, reduced_T_grid)
    T_array = T_grid_mesh.flatten()
    rho_array = rho_grid_mesh.flatten()

    mictherm_names, mictherm_units, mictherm_values, InputT, Inputp, Inputrho, Inputx = mictherm_grid(
        mode="userproperties",
        T=T_array * factorTc,
        rho=rho_array * factorRho,
        p=None,
        x=np.ones_like(T_array),
        print_output=False,
        base_user_parameters=MIC_THERM_USER_PARAMETERS,
    )
    mictherm_properties = extract_mictherm_properties(
        MIC_THERM_USER_PARAMETERS["properties"],
        mictherm_values,
    )
    grid_shape = (reduced_T_grid.shape[0], reduced_rho_grid.shape[0])
    grid = {
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


if __name__ == "__main__":
    # interactive selection: use existing precomputed grids (debugging=1)
    # or generate new grids (debugging=0). User can adjust Tr in both cases.
    def _inp(prompt, default):
        v = input(f"{prompt} [{default}]: ")
        return v.strip() or str(default)

    print("Select grid mode:\n 1) Use existing precomputed grid (faster)\n 2) Generate new grid (full recompute)")
    mode = _inp("Choose 1 or 2", "1")
    debugging = 1 if mode == "1" else 0

    Tr = float(_inp("Reduced temperature Tr", 0.85))

    # When using precomputed grids, user should check Tmin/Tstep/rhostep are as desired.
    # When generating new grids (debugging==0), allow changing these parameters here.
    Trmin = 0.3
    Tstep = 0.01
    rhostep = 0.03
    print(f"Current grid defaults: Trmin={Trmin}, Tstep={Tstep}, rhostep={rhostep}.")
    print("If these are not suitable, regenerate the grid (mode 2); MicTherm runtime is required.")
    if debugging == 0:
        Trmin = float(_inp("Trmin (min reduced temperature)", Trmin))
        Tstep = float(_inp("Tstep (temperature step)", Tstep))
        rhostep = float(_inp("rhostep (density step)", rhostep))
    else:
        print(f"Using precomputed grids. Current Trmin={Trmin}, Tstep={Tstep}, rhostep={rhostep}.")
        print("To change these values, choose mode 2 (Generate new grid), because regeneration requires MicTherm runtime.")
    # create a top-level 'temp' folder in the repository root and use it for temporary files
    repo_root = Path(__file__).resolve().parents[4]
    temp_dir = repo_root / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "temp_mictherm_grids_CO2.npz"

    if debugging == 0:
        generate_grid(temp_file, Trmin=Trmin, Tstep=Tstep, rhostep=rhostep)

    if not temp_file.exists():
        raise FileNotFoundError(
            f"{temp_file} does not exist. Run once with grid mode 2 "
            "to calculate and cache the MicTherm data."
        )

    # Load all critical-point, VLE, and grid data (reduced units, see generate_grid).
    with np.load(temp_file) as data:
        Tc = data["Tc"].item()
        rhoc = data["rhoc"].item()
        pc = data["pc"].item()
        micthermVLE_T = data["VLE_T"]
        micthermVLE_rho_l = data["VLE_rho_l"]
        micthermVLE_rho_g = data["VLE_rho_g"]
        T_grid = data["T_grid"]
        rho_grid = data["rho_grid"]
        p_grid = data["p_grid"]
        eta_grid = data["eta_grid"]
        gamma_surface_grid = data["gamma_surface_grid"]

    T = Tr * Tc
    interp_rho_l = interp1d(micthermVLE_T, micthermVLE_rho_l, kind="cubic", fill_value="extrapolate")
    interp_rho_g = interp1d(micthermVLE_T, micthermVLE_rho_g, kind="cubic", fill_value="extrapolate")
    rho_l = interp_rho_l(T)
    rho_g = interp_rho_g(T)

    # Critical point + VLE + grids (cache units) for the VLE overlay in the pressure plots.
    grid = {"Tc": Tc, "rhoc": rhoc, "pc": pc, "VLE_T": micthermVLE_T, "VLE_rho_l": micthermVLE_rho_l,
            "VLE_rho_g": micthermVLE_rho_g, "T_grid": T_grid, "rho_grid": rho_grid, "p_grid": p_grid,
            "eta_grid": eta_grid, "gamma_surface_grid": gamma_surface_grid}

    # Save the pressure grid once as its own publication-style figure.
    # Physical quantities are italic; units are upright.
    pressure_fig = plt.figure(figsize=(7, 5))
    pressure_ax = pressure_fig.add_subplot(111, projection="3d")
    pressure_ax.plot_surface(
        rho_grid, T_grid, p_grid, cmap=cm.nipy_spectral, alpha=0.5
    )
    plot_vle_on_pressure_grid(pressure_ax, grid)
    pressure_ax.set_xlabel(r"$\rho^*$")
    pressure_ax.set_ylabel(r"$T^*$")
    pressure_ax.set_zlabel(r"$p^*$")
    pressure_ax.set_title("MicTherm pressure grid (3D)")

    pressure_grid_dir = repo_root / "output"
    pressure_grid_dir.mkdir(parents=True, exist_ok=True)
    pressure_grid_path = pressure_grid_dir / "mictherm_pressure_grid_CO2.png"
    pressure_fig.tight_layout()
    pressure_fig.savefig(pressure_grid_path, dpi=300, bbox_inches="tight")
    print(f"Saved pressure-grid plot to {pressure_grid_path}")
    plt.show()
    plt.close(pressure_fig)

    fig = plt.figure(figsize=(16, 5))
    
    # Subplot 1: Pressure grid
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(rho_grid, T_grid, p_grid, cmap=cm.nipy_spectral, alpha=0.5)
    plot_vle_on_pressure_grid(ax1, grid, verbose=False)
    ax1.set_xlabel(r"$\rho$")
    ax1.set_ylabel(r"$T$")
    ax1.set_zlabel(r"$p$")
    ax1.set_title("MicTherm pressure grid (3D)")
    
    # Subplot 2: Viscosity grid
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(rho_grid, T_grid, eta_grid, cmap=cm.viridis, alpha=0.8)
    ax2.set_xlabel(r"$\rho$")
    ax2.set_ylabel(r"$T$")
    ax2.set_zlabel(r"$\eta$")
    ax2.set_title("MicTherm viscosity grid (3D)")
    
    # Subplot 3: Surface tension grid
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(rho_grid, T_grid, gamma_surface_grid, cmap=cm.plasma, alpha=0.8)
    ax3.set_xlabel(r"$\rho$")
    ax3.set_ylabel(r"$T$")
    ax3.set_zlabel(r"$\gamma_{surface}$")
    ax3.set_title("MicTherm surface tension grid (3D)")

    plt.tight_layout()
    plt.show()

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

    r = 30
    nx = 200
    ny = 200

    width = 3

    T_l = 0.9 * T
    T_g = 1.1 * T
    x = np.linspace(0, nx - 1, nx, dtype=int)
    y = np.linspace(0, ny - 1, ny, dtype=int)
    x, y = np.meshgrid(x, y)
    # Vertical temperature profile: top (y=0) is hotter (T_g), bottom (y=ny-1) is colder (T_l)
    # normalized vertical coordinate (0 at top, 1 at bottom)
    y_norm = np.linspace(0.0, 1.0, ny, dtype=float)
    T_field = np.tile(T_l + (T_g - T_l) * y_norm, (nx, 1))
    T_field = T_field.reshape((nx, ny, 1))

    # T_field = 0.5 * (T_l + T_g) - 0.5 * (T_l - T_g) * np.tanh(2 * (dist - r) / width)
    # T_field = T_field.reshape((nx, ny, 1))

    a = 9 / 49
    b = 2 / 21
    R = 1.0


    s_rho = [0.0]
    s_e = [1.2]
    s_eta = [1.0]
    s_j = [0.0]
    s_q = [1.0]
    s_v = [1.0]

    
    # Note for Martin: I don't know on which position you want to use eta_grid and gamma_surface_grid, so I just saved them in kwargs for now. You can access them later in the code as needed.
    # for rho,p,T, the grid intepolation has been carried out in MicTherm class, so you can access them directly from the eos object.
    kwargs = {
        "rho_grid": rho_grid, # unit: reduced
        "p_grid": p_grid, # the p_grid has been saved here in kwargs for later use in the further code, Unit: reduced
        "T_grid": T_grid, # unit: reduced
        "eta_grid": eta_grid,  # the eta_grid has been saved here in kwargs for later use in the further code, Unit: reduced
        "gamma_surface_grid": gamma_surface_grid, # the gamma_surface has been saved here in kwargs for later use in the further code, Unit: reduced
    }
    eos = MicTherm(**kwargs)
    # example of how to interpolate properties at a specific state point (rho=12.5, T=280.0)
    properties = interpolate_mictherm_properties(
        kwargs=kwargs,
        rho=0.8,
        T=0.9,
        )

    p = properties["p"]
    eta = properties["eta"]
    gamma_surface = properties["gamma_surface"]

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
        "io_rate": 10,
        "compute_MLUPS": False,
        "print_info_rate": 10,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }

    sim = Droplet2D(**kwargs)
    sim.run(200)
