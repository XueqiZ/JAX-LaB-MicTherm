"""
MicTherm PeTS droplet transported through a two-dimensional channel.

The x direction is periodic, while the top and bottom boundaries are stationary
no-slip walls (full-way bounce-back). A positive-x body force transports the
droplet through a long channel. The liquid and vapor coexistence densities and
the EOS table are supplied by MicTherm.

The collision matrix is based on:
1. McCracken, M. E. & Abraham, J. Multiple-relaxation-time lattice-Boltzmann model for multiphase flow. Phys. Rev. E 71, 036701 (2005).
"""

import os
import sys
from jax import config
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MicLBM_src import *
from src.lattice import LatticeD2Q9
from src.boundary_conditions import BounceBack
from MicLBM_src.eos import MicTherm, interpolate_mictherm_properties
from src.utils import *
from MicLBM_src.Mic_utils import save_fields_vtk as save_mictherm_fields_vtk
from MicLBM_src.Mic_multiphase import MultiphaseMRTTvar


# config.update("jax_default_matmul_precision", "float32")


class DropletChannel2D(MultiphaseMRTTvar):
    def initialize_macroscopic_fields(self):
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        x, y = np.meshgrid(x, y, indexing="ij")

        rho_tree = []

        # Start away from the periodic edges to delay interaction with its
        # periodic image.
        dist = np.sqrt((x - droplet_center_x) ** 2 + (y - droplet_center_y) ** 2)
        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(
            2 * (dist - droplet_radius) / interface_width
        )

        rho = rho.reshape((self.nx, self.ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        """Apply stationary walls; left and right remain periodic."""
        walls = np.concatenate(
            (self.boundingBoxIndices["top"], self.boundingBoxIndices["bottom"])
        )
        walls = tuple(walls.T)
        self.BCs[0].append(
            BounceBack(
                walls,
                self.gridInfo,
                self.precisionPolicy,
                theta=theta[walls],
                phi=phi[walls],
                delta_rho=delta_rho[walls],
            )
        )

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        p = np.array(kwargs["p"][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        T_field = np.array(self.T_field)
        fields = {
            "p": p[..., 0],
            "rho": rho[..., 0],
            "T": T_field[..., 0],
            "velocity": u,
            "velocity_magnitude": np.linalg.norm(u, axis=-1),
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        print(f"Density field: Min: {np.min(rho):.8g} Max: {np.max(rho):.8g}")
        print(f"Maxwell construction: Min: {rho_g} Max: {rho_l}")
        print(f"Spurious currents: {np.max(np.sqrt(np.sum(u**2, axis=-1)))}")

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        save_mictherm_fields_vtk(timestep, fields, output_dir, "data")
        save_image(timestep, u)
        

if __name__ == "__main__":

    # Load a precomputed MicTherm grid and VLE curve.
    repo_root = Path(__file__).resolve().parents[4]
    temp_dir = repo_root / "temp"
    candidates = sorted(temp_dir.glob("temp_mictherm_grids*.npz")) if temp_dir.exists() else []
    if not candidates:
        raise FileNotFoundError(
            f"No temp_mictherm_grids*.npz found in {temp_dir}. "
            "Generate one first with droplet_2d_2D_Mic_PeTS.py."
        )
    if len(candidates) == 1:
        temp_file = candidates[0]
    else:
        print("Available temp files:")
        for i, path in enumerate(candidates, start=1):
            print(f"{i}: {path.name}")
        try:
            choice = int(input(f"Select file [1-{len(candidates)}]: ").strip())
            if not 1 <= choice <= len(candidates):
                raise ValueError
        except Exception:
            print("Invalid selection, using first file.")
            choice = 1
        temp_file = candidates[choice - 1]

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

    
    fig = plt.figure(figsize=(16, 5))
    
    # Subplot 1: Pressure grid
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(rho_grid, T_grid, p_grid, cmap=cm.nipy_spectral, alpha=0.8)
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

    # simulation domain size: 
    nx = 400
    ny = 100

    # Droplet and diffuse-interface geometry.  Keep the interface several
    # lattice nodes away from the no-slip walls.
    droplet_center_x = 0.30 * nx
    droplet_center_y = 0.50 * (ny - 1)
    droplet_radius = 30
    interface_width = 6

    # Wall pseudo-density parameters. theta=pi/2 gives a neutral 90-degree
    # contact angle if the droplet later reaches a wall.
    theta = (np.pi / 2) * np.ones((nx, ny, 1))
    phi = np.ones((nx, ny, 1))
    delta_rho = np.zeros((nx, ny, 1))

    try:
        Tr = float(input("Define reduced base temperature Tr (e.g. 0.85): "))
    except Exception:
        Tr = 0.85
    T = Tr * Tc

    temperature_initialization = input(
        "Select temperature initialization ('constant' or 'gradient'): "
    ).strip().lower()
    if temperature_initialization not in ("constant", "gradient"):
        temperature_initialization = "constant"

    if temperature_initialization == "constant":
        constant_temperature = T
        T_left = T
        T_right = T
    else:
        try:
            t_left_mul = float(input("Define left-boundary temperature multiplier (e.g. 1.1): "))
        except Exception:
            t_left_mul = 1.1
        try:
            t_right_mul = float(input("Define right-boundary temperature multiplier (e.g. 0.9): "))
        except Exception:
            t_right_mul = 0.9
        T_left = t_left_mul * T
        T_right = t_right_mul * T

    interp_rho_l = interp1d(
        micthermVLE_T,
        micthermVLE_rho_l,
        kind="cubic",
        fill_value="extrapolate",
    )
    interp_rho_g = interp1d(
        micthermVLE_T,
        micthermVLE_rho_g,
        kind="cubic",
        fill_value="extrapolate",
    )
    rho_l = interp_rho_l(T)
    rho_g = interp_rho_g(T)

    x = np.linspace(0, nx - 1, nx, dtype=int)
    y = np.linspace(0, ny - 1, ny, dtype=int)
    x, y = np.meshgrid(x, y)

    if temperature_initialization == "constant":
        T_field = np.full((nx, ny), constant_temperature, dtype=float)
    elif temperature_initialization == "gradient":
        # Linear channel-wise profile: left at x=0, right at x=nx-1.
        x_profile = np.linspace(T_left, T_right, nx, dtype=float)
        T_field = np.repeat(x_profile[:, np.newaxis], ny, axis=1)
    else:
        raise ValueError(
            "temperature_initialization must be 'constant' or 'gradient', "
            f"not {temperature_initialization!r}"
        )

    T_field = T_field.reshape((nx, ny, 1))

    s_rho = [0.0]
    s_e = [1.2]
    s_eta = [1.0]
    s_j = [0.0]
    s_q = [1.0]
    s_v = [1.0]


    kwargs = {
        "rho_grid": rho_grid, # unit: reduced density
        "p_grid": p_grid, # the p_grid has been saved here in kwargs for later use in the further code, Unit: reduced pressure
        "T_grid": T_grid, # unit: 'K'
        "eta_grid": eta_grid,  # the eta_grid has been saved here in kwargs for later use in the further code, Unit: reduced viscosity
        "gamma_surface_grid": gamma_surface_grid, # the gamma_surface has been saved here in kwargs for later use in the further code, Unit: reduced surface tension
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
        # Positive x drives the fluid toward the open right outlet. This is an
        # acceleration-like force; MultiphaseMRTTvar multiplies it by rho.
        "body_force": [5.0e-6, 0.0],
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
        "io_rate": 50,
        "compute_MLUPS": False,
        "print_info_rate": 50,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }

    sim = DropletChannel2D(**kwargs)
    steps = input("How many steps should be run after starting the simulation? [3000]: ")
    steps = int(steps) if steps.strip() else 3000
    sim.run(steps)
