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
from MicLBM_src.eos import MicTherm
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
        "units": "SI",
        "Output": "no",
        "Debug": "no",
        "stability": "no",

        # EOS / model settings
        "N_components": 2,
        "Substance_ID1": 103,
        "PotModel_1": "LJ.pm",
        "Substance_ID2": 0,
        "PotModel_2": "LJ.pm",
        "EOS": "PC_SAFT",
        #"IDEAL": "IdealQM", #current not nessary for rho 

        # Substance parameters - Component 2 (R-1234yf)
        "epsilon_2": 179.953,
        "sigma_2": 3.631,
        "chainlength_2": 2.358,
        "molar_mass_2": 114.04,
        #"CAS_number_2": "754-12-1",
        "Polar_2": "GrossVrabec",
        "Dipolemoment_2": 2.011,
        "Dipolemoment_xp_2": 0.33,
        "properties": "p",
}

if __name__ == "__main__":
    # Interactive selection: use an existing precomputed grid (debugging=1)
    # or generate a new grid (debugging=0).
    def _inp(prompt, default):
        v = input(f"{prompt} [{default}]: ")
        return v.strip() or str(default)

    print("Select grid mode:\n 1) Use existing precomputed grid (faster)\n 2) Generate new grid (full recompute)")
    mode = _inp("Choose 1 or 2", "1")
    debugging = 1 if mode == "1" else 0

    T_iso = float(_inp("VLE isotherm temperature T_iso", 300))
    x_composition = float(
        _inp("Composition x (mole fraction of component 1, CO2)", 1.0)
    )
    if not 0.0 <= x_composition <= 1.0:
        raise ValueError("Composition x must be between 0 and 1")
    print(
        f"Calculating x_CO2={x_composition} and "
        f"x_HFO-1234yf={1.0 - x_composition}."
    )

    # These are absolute MicTherm temperatures and densities; no critical
    # temperature or critical density is needed to construct the grid.
    Tmin = 270.0
    Tmax = 360.0
    Tstep = 100
    rhomin = 0.5
    rhomax = 10
    rhostep = 0.5
    if debugging == 0:
        Tmin = float(_inp("Tmin (minimum absolute temperature)", Tmin))
        Tmax = float(_inp("Tmax (maximum absolute temperature)", Tmax))
        Tstep = float(_inp("Tstep (absolute temperature step)", Tstep))
        rhomin = float(_inp("rhomin (minimum absolute density)", rhomin))
        rhomax = float(_inp("rhomax (maximum absolute density)", rhomax))
        rhostep = float(_inp("rhostep (absolute density step)", rhostep))
        if Tmin >= Tmax or rhomin >= rhomax:
            raise ValueError("Tmin must be below Tmax and rhomin must be below rhomax")
        if Tstep <= 0.0 or rhostep <= 0.0:
            raise ValueError("Tstep and rhostep must be positive")
    else:
        print("Using the bounds stored in the precomputed grid.")
        print("To change the bounds, choose mode 2 (Generate new grid).")
    # create a top-level 'temp' folder in the repository root and use it for temporary files
    repo_root = Path(__file__).resolve().parents[4]
    temp_dir = repo_root / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "temp_mictherm_grids_HFO-1234yf-CO2.npz"

    if debugging == 1:
        if not temp_file.exists():
            raise FileNotFoundError(
                f"{temp_file} does not exist. Run once with debugging = 0 "
                "to calculate and cache the MicTherm data."
            )

        # Load the precomputed isothermal VLE curve and property grids.
        with np.load(temp_file) as data:
            cached_x = data["x_composition"].item() if "x_composition" in data else 1.0
            if "VLE_x_l" not in data or "T_iso" not in data:
                raise ValueError(
                    "This cache uses the old temperature-based VLE format. "
                    "Choose mode 2 to regenerate it with xL-based interpolation."
                )
            cached_T_iso = data["T_iso"].item()
            micthermVLE_x_l = data["VLE_x_l"]
            micthermVLE_rho_l = data["VLE_rho_l"]
            micthermVLE_rho_g = data["VLE_rho_g"]
            T_grid = data["T_grid"]
            rho_grid = data["rho_grid"]
            p_grid = data["p_grid"]

        if not np.isclose(x_composition, cached_x):
            raise ValueError(
                f"The cache was calculated at x_CO2={cached_x}, but "
                f"x_CO2={x_composition} was requested. Choose mode 2 to "
                "generate a grid for this composition."
            )
        if not np.isclose(T_iso, cached_T_iso):
            raise ValueError(
                f"The cache contains VLE data at T_iso={cached_T_iso}, but "
                f"T_iso={T_iso} was requested. Choose mode 2 to regenerate it."
            )

        T = T_iso
    else:
        T = T_iso

        # Calculate only the requested isothermal VLE point.
        mictherm_names, mictherm_units, mictherm_values = run_mictherm_func(
            mode="VLE_Iso",
            T=None,
            rho=None,
            p=None,
            x=x_composition,
            t_iso=T_iso,
            init_mode="uninitialized",
            print_output=False,
            base_user_parameters=MIC_THERM_USER_PARAMETERS,
        )
        micthermVLE_x_l = mictherm_values[:, 1]
        micthermVLE_rho_l = mictherm_values[:, 5]
        micthermVLE_rho_g = mictherm_values[:, 6]

        # Create meshgrid to combine rho and T dimensions.
        mictherm_T_grid = np.arange(Tmin, Tmax + 0.5 * Tstep, Tstep)
        mictherm_rho_grid = np.arange(rhomin, rhomax + 0.5 * rhostep, rhostep)
        rho_grid_mesh, T_grid_mesh = np.meshgrid(mictherm_rho_grid, mictherm_T_grid)
        T_array = T_grid_mesh.flatten()
        rho_array = rho_grid_mesh.flatten()

        # Generate array and calculate mictherm_grid
        mictherm_names, mictherm_units, mictherm_values, InputT, Inputp, Inputrho, Inputx = mictherm_grid(
            mode="userproperties",
            T=T_array,
            rho=rho_array,
            p=None,
            x=np.full_like(T_array, x_composition, dtype=float),
            print_output=False,
            base_user_parameters=MIC_THERM_USER_PARAMETERS,
        )
        mictherm_properties = extract_mictherm_properties(
            MIC_THERM_USER_PARAMETERS["properties"],
            mictherm_values,
        )
        T_grid = InputT.reshape(mictherm_T_grid.shape[0], mictherm_rho_grid.shape[0])
        rho_grid = Inputrho.reshape(mictherm_T_grid.shape[0], mictherm_rho_grid.shape[0])
        p_grid = mictherm_properties["p"].reshape(mictherm_T_grid.shape[0], mictherm_rho_grid.shape[0])
        

        np.savez(
            temp_file,
            x_composition=x_composition,
            T_iso=T_iso,
            VLE_x_l=micthermVLE_x_l,
            VLE_rho_l=micthermVLE_rho_l,
            VLE_rho_g=micthermVLE_rho_g,
            T_grid=T_grid,
            rho_grid=rho_grid,
            p_grid=p_grid,
        )

    # At fixed T_iso, interpolate both coexistence densities as functions of
    # the liquid CO2 mole fraction xL.
    vle_sort = np.argsort(micthermVLE_x_l)
    vle_x_l = np.asarray(micthermVLE_x_l)[vle_sort]
    vle_rho_l = np.asarray(micthermVLE_rho_l)[vle_sort]
    vle_rho_g = np.asarray(micthermVLE_rho_g)[vle_sort]
    vle_x_l, unique_indices = np.unique(vle_x_l, return_index=True)
    vle_rho_l = vle_rho_l[unique_indices]
    vle_rho_g = vle_rho_g[unique_indices]
    if vle_x_l.size < 2:
        raise ValueError("At least two distinct xL values are required for VLE interpolation")
    interpolation_kind = "cubic" if vle_x_l.size >= 4 else "linear"
    interp_rho_l = interp1d(
        vle_x_l, vle_rho_l, kind=interpolation_kind, fill_value="extrapolate"
    )
    interp_rho_g = interp1d(
        vle_x_l, vle_rho_g, kind=interpolation_kind, fill_value="extrapolate"
    )
    rho_l = float(interp_rho_l(x_composition))
    rho_g = float(interp_rho_g(x_composition))

    

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(rho_grid, T_grid, p_grid, cmap=cm.nipy_spectral, alpha=0.8)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$T$")
    ax.set_zlabel(r"$p$")
    ax.set_title("MicTherm pressure grid (3D)")

    plt.tight_layout()
    plt.show()

    # scale back by factorPc;
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

    
    # Pressure is interpolated over the (T, rho) grid by the MicTherm EOS.
    kwargs = {
        "rho_grid": rho_grid, # unit: 'mol/L'
        "p_grid": p_grid, # the p_grid has been saved here in kwargs for later use in the further code, Unit: 'MPa'
        "T_grid": T_grid, # unit: 'K'
    }
    eos = MicTherm(**kwargs)

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
