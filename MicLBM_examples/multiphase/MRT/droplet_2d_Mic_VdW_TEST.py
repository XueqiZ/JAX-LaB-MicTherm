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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
from src.utils import *
from MicLBM_src.Mic_multiphase import MultiphaseMRT


# config.update("jax_default_matmul_precision", "float32")

# --- global parameters ---
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

width = 5

a = 9 / 49
b = 2 / 21
R = 1.0

rho_l = 10.0
rho_g = 0.00001

Tc = 0.5714285714

s_rho = [0.0]
s_e = [1.2]
s_eta = [1.0]
s_j = [0.0]
s_q = [1.0]
s_v = [1.0]

class Droplet2D(MultiphaseMRT):
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
    
    
    def find_diameter_along_line(self,line,rho_mid):
   
        x1 = None
        x2 = None

        # scan for crossings
        for i in range(len(line) - 1):

            r0 = line[i]
            r1 = line[i + 1]

            # ---- first crossing: below -> above ----
            if x1 is None and (r0 < rho_mid and r1 >= rho_mid):
                # linear interpolation
                t = (rho_mid - r0) / (r1 - r0)
                x1 = i + t

            # ---- second crossing: above -> below ----
            elif x1 is not None and (r0 >= rho_mid and r1 < rho_mid):
                t = (rho_mid - r0) / (r1 - r0)
                x2 = i + t
                break

        if None in (x1, x2):
            return None  # droplet not detected properly

        return x2 - x1
    
    def droplet_diameter(self, rho, x, y, rho_l, rho_g, offset=5):

        rho_mid = 0.5 * (rho_l + rho_g)

        diameters_x = []
        diameters_y = []

        # sweep along y-direction (horizontal lines)
        for dy in range(-offset, offset + 1):
            yy = y + dy
            if yy < 0 or yy >= rho.shape[1]:
                continue

            line = rho[:, yy]
            d = self.find_diameter_along_line(line, rho_mid)

            if d is not None:
                diameters_x.append(d)

        # sweep along x-direction (vertical lines)
        for dx in range(-offset, offset + 1):
            xx = x + dx
            if xx < 0 or xx >= rho.shape[0]:
                continue

            line = rho[xx, :]
            d = self.find_diameter_along_line(line, rho_mid)

            if d is not None:
                diameters_y.append(d)

        if len(diameters_x) == 0 or len(diameters_y) == 0:
            return None

        # take largest diameter
        return 0.5*(max(diameters_x)+max(diameters_y))
    
    def interface_thickness(self, rho, x, y, rho_l, rho_g, rho_offset=0.1):

        rho_l_offset = rho_l - rho_offset*(rho_l-rho_g)
        rho_g_offset = rho_g + rho_offset*(rho_l-rho_g)

        # interface thickness (horizontal lines)
        line = rho[:, y]
        d_l_x = self.find_diameter_along_line(line, rho_l_offset)
        d_g_x = self.find_diameter_along_line(line, rho_g_offset)
        d_x = 0.5*(d_g_x - d_l_x)

        # interface thickness (vertical lines)
        line = rho[x, :]
        d_l_y = self.find_diameter_along_line(line, rho_l_offset)
        d_g_y = self.find_diameter_along_line(line, rho_g_offset)
        d_y = 0.5*(d_g_y - d_l_y)

        if d_x == 0 or d_y == 0:
            return None

        # take largest diameter
        return 0.5*(d_x+d_y)

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)      
        self.last_rho = np.array(kwargs["rho_tree"][0][0, ...])
        self.last_u = np.array(kwargs["u_tree"][0][0, ...])

        rho = self.last_rho
        u = self.last_u
        p = np.array(kwargs["p"][0, ...])
        
        timestep = kwargs["timestep"]
        fields = {"p": p[..., 0], "rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1]}
        offset = 90
        rho_north = rho[self.nx // 2, self.ny // 2 - offset, 0]
        rho_south = rho[self.nx // 2, self.ny // 2 + offset, 0]
        rho_west = rho[self.nx // 2 - offset, self.ny // 2, 0]
        rho_east = rho[self.nx // 2 + offset, self.ny // 2, 0]
        rho_g_pred = 0.25 * (rho_north + rho_south + rho_west + rho_east)
        rho_l_pred = rho[self.nx // 2, self.ny // 2, 0]
        print(f"k: {self.k}, A: {self.A}")
        print(f"%Error Min: {(rho_g_pred - rho_g) * 100 / rho_g} Max: {(rho_l_pred - rho_l) * 100 / rho_l}")
        print(f"Density: Min: {rho_g_pred} Max: {rho_l_pred}")
        print(f"Maxwell construction: Min: {rho_g} Max: {rho_l}")
        print(f"Spurious currents: {np.max(np.sqrt(np.sum(u**2, axis=-1)))}")
        p_north = p[self.nx // 2, self.ny // 2 - offset, 0]
        p_south = p[self.nx // 2, self.ny // 2 + offset, 0]
        p_west = p[self.nx // 2 - offset, self.ny // 2, 0]
        p_east = p[self.nx // 2 + offset, self.ny // 2, 0]
        pressure_difference = p[self.nx // 2, self.ny // 2, 0] - 0.25 * (p_north + p_south + p_west + p_east)
        radius = 0.5 * self.droplet_diameter(rho[:,:,0], self.nx // 2, self.ny // 2, rho_l, rho_g)
        interface = self.interface_thickness(rho[:,:,0], self.nx // 2, self.ny // 2, rho_l, rho_g)
        surfTens = np.abs(pressure_difference*radius)
        self.last_int_thick = interface
        self.last_radius = radius
        self.last_surfTens = surfTens
        print(f"Pressure difference: {pressure_difference}, Radius: {radius}, surfTens: {surfTens}, interThick: {interface}")
        #save_fields_vtk(timestep, fields, "output", "data")
        
MIC_THERM_USER_PARAMETERS = {
        # General
        "units": "SI",
        "Output": "no",
        "Debug": "no",
        "stability": "no",

        "N_components": 1,
        "Substance_ID1": 0,
        "PotModel_1": "LJ.pm",
        "units": "SI",
        "EOS": "vanderWaals",
        "Output": "no",
        "Debug": "no",
        "chainlength_1": 1,
        "b_VDW_1": 0.0952,
        "a_VDW_1": 0.1837,
        "molar_mass_1": 114.04,
        "eta_param": "0,-16.887065518,15.000907502,-2.8455190264,0.44056790447", #data from Argon, didn't have any physical meaning, just to get a reasonable viscosity value for the simulation
        "DGT_kappa_1": 6.738872, #ditto
        "transPropMode": "entropyScaling",
        "properties": "p, eta, gamma_surface",
}

def run_simulation(T_X, kappa, k_val, A_val, rho_l_local, rho_g_local, steps=10000):
    global rho_l, rho_g
    rho_l = rho_l_local
    rho_g = rho_g_local

    T = T_X * Tc
	# set debugging: 1 -> load/save temp grids to speed up debugging, 0 -> full recompute
    debugging = 0
    # create a top-level 'temp' folder in the repository root and use it for temporary files
    repo_root = Path(__file__).resolve().parents[3]
    temp_dir = repo_root / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "temp_mictherm_grids_VdW.npz"
    # calculate critical point properties using MicTherm API
    mictherm_names, mictherm_units, mictherm_values = run_mictherm_func(
        mode="criticalpoint",
        T=None,
        rho=None,
        p=None,
        x=None,
        t_iso=None,
        init_mode="uninitialized",
        print_output=False,
        base_user_parameters=MIC_THERM_USER_PARAMETERS,
    )

    # reference: publication 
    Tc = mictherm_values[0,0]
    rhoc = mictherm_values[0,1]
    pc = mictherm_values[0,2]

    Tc = 4/7
    rhoc = 7/2
    pc = (9/49) / (27 * (2/21) ** 2)

    factorTc = mictherm_values[0,0]/Tc
    factorRho = mictherm_values[0,1]/rhoc
    factorPc = mictherm_values[0,2]/pc

    # calculate VLE properties using MicTherm API for Tiso
    mictherm_names, mictherm_units, mictherm_values = run_mictherm_func(
            mode="vle_iso",
            t_iso=T * factorTc,  # scale Tiso by factorTc to be consistent with critical point properties
            T=None,
            rho=None,
            p=None,
            x=1,
            init_mode="uninitialized",
            print_output=False,
            base_user_parameters=MIC_THERM_USER_PARAMETERS,
    )
    mictherm_rho_l = mictherm_values[0, 1] 
    mictherm_rho_g = mictherm_values[0, 2]
    mictherm_T = T * factorTc  # scale T by factorTc to be consistent with critical point properties
    rho_l = mictherm_rho_l / factorRho  # scale back by factorRho
    rho_g = mictherm_rho_g / factorRho  # scale back by factorRho

    # Create meshgrid to combine rho and T dimensions
    rho_step = 10
    T_step = 10
    mictherm_T_grid = np.linspace(mictherm_T * 0.95,mictherm_T * 1.1, T_step)
    mictherm_rho_grid = np.linspace(mictherm_rho_g * 0.85,  mictherm_rho_l * 1.3, rho_step)

    if debugging and temp_file.exists():
        # load precomputed grids
        data = np.load(temp_file)
        T_grid = data["T_grid"]
        rho_grid = data["rho_grid"]
        p_grid = data["p_grid"]
        eta_grid = data["eta_grid"]
        gamma_surface_grid = data["gamma_surface_grid"]
    else:
        rho_grid_mesh, T_grid_mesh = np.meshgrid(mictherm_rho_grid, mictherm_T_grid)
        T_array = T_grid_mesh.flatten()
        rho_array = rho_grid_mesh.flatten()

        # Generate array and calculate mictherm_grid
        mictherm_names, mictherm_units, mictherm_values, InputT, Inputp, Inputrho, Inputx = mictherm_grid(
            mode="userproperties",
            T=T_array,
            rho=rho_array,
            p=None,
            x=np.ones_like(T_array),
            print_output=False,
            base_user_parameters=MIC_THERM_USER_PARAMETERS,
        )
        mictherm_properties = extract_mictherm_properties(
            MIC_THERM_USER_PARAMETERS["properties"],
            mictherm_values,
        )
        T_grid = InputT.reshape(mictherm_T_grid.shape[0], mictherm_rho_grid.shape[0]) / factorTc  # scale back by factorTc
        rho_grid = Inputrho.reshape(mictherm_T_grid.shape[0], mictherm_rho_grid.shape[0]) / factorRho  # scale back by factorRho
        p_grid = mictherm_properties["p"].reshape(mictherm_T_grid.shape[0], mictherm_rho_grid.shape[0]) / factorPc  # scale back by factorPc
        eta_grid = mictherm_properties["eta"].reshape(mictherm_T_grid.shape[0], mictherm_rho_grid.shape[0])
        gamma_surface_grid = mictherm_properties["gamma_surface"].reshape(mictherm_T_grid.shape[0], mictherm_rho_grid.shape[0])

        if debugging == 0:
            np.savez(temp_file, T_grid=T_grid, rho_grid=rho_grid, p_grid=p_grid, eta_grid=eta_grid, gamma_surface_grid=gamma_surface_grid)

    kwargs = {
        "rho_grid": rho_grid, # unit: 'mol/L'
        "p_grid": p_grid, # the p_grid has been saved here in kwargs for later use in the further code, Unit: 'MPa'
        "T_grid": T_grid, # unit: 'K'
        "eta_grid": eta_grid,  # the eta_grid has been saved here in kwargs for later use in the further code, Unit: 'Pa.s'
        "gamma_surface_grid": gamma_surface_grid, # the gamma_surface has been saved here in kwargs for later use in the further code, Unit: 'J/m^2'
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
        "body_force": [0.0, 0.0],
        "k": [k_val],                    #"k": [0.16],
        "A": -1 *(A_val)* np.ones((1, 1)),  #"A": -0.032 * np.ones((1, 1)),
        "M": [M],
        "s_rho": s_rho,
        "s_e": s_e,
        "s_eta": s_eta,
        "s_j": s_j,
        "s_q": s_q,
        "s_v": s_v,
        "kappa": [kappa],
        "precision": precision,
        "io_rate": steps,
        "compute_MLUPS": False,
        "print_info_rate": 10000,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }

    sim = Droplet2D(**kwargs)
    sim.run(steps)

    return sim.last_rho, sim.last_u, sim.last_int_thick, sim.last_radius, sim.last_surfTens

if __name__ == "__main__":
    # set debugging: 1 -> load/save temp grids to speed up debugging, 0 -> full recompute
    debugging = 0
    # create a top-level 'temp' folder in the repository root and use it for temporary files
    repo_root = Path(__file__).resolve().parents[3]
    temp_dir = repo_root / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "temp_mictherm_grids_VdW.npz"
    # calculate critical point properties using MicTherm API
    mictherm_names, mictherm_units, mictherm_values = run_mictherm_func(
        mode="criticalpoint",
        T=None,
        rho=None,
        p=None,
        x=None,
        t_iso=None,
        init_mode="uninitialized",
        print_output=False,
        base_user_parameters=MIC_THERM_USER_PARAMETERS,
    )

    # reference: publication 
    Tc = mictherm_values[0,0]
    rhoc = mictherm_values[0,1]
    pc = mictherm_values[0,2]

    T = 0.85* Tc

    Tc = 4/7
    rhoc = 7/2
    pc = (9/49) / (27 * (2/21) ** 2)

    T = 0.8 * Tc

    factorTc = mictherm_values[0,0]/Tc
    factorRho = mictherm_values[0,1]/rhoc
    factorPc = mictherm_values[0,2]/pc

    # calculate VLE properties using MicTherm API for Tiso
    mictherm_names, mictherm_units, mictherm_values = run_mictherm_func(
            mode="vle_iso",
            t_iso=T * factorTc,  # scale Tiso by factorTc to be consistent with critical point properties
            T=None,
            rho=None,
            p=None,
            x=1,
            init_mode="uninitialized",
            print_output=False,
            base_user_parameters=MIC_THERM_USER_PARAMETERS,
    )
    mictherm_rho_l = mictherm_values[0, 1] 
    mictherm_rho_g = mictherm_values[0, 2]
    mictherm_T = T * factorTc  # scale T by factorTc to be consistent with critical point properties
    rho_l = mictherm_rho_l / factorRho  # scale back by factorRho
    rho_g = mictherm_rho_g / factorRho  # scale back by factorRho

    # Create meshgrid to combine rho and T dimensions
    rho_step = 10
    T_step = 10
    mictherm_T_grid = np.linspace(mictherm_T * 0.95,mictherm_T * 1.1, T_step)
    mictherm_rho_grid = np.linspace(mictherm_rho_g * 0.85,  mictherm_rho_l * 1.3, rho_step)

    if debugging and temp_file.exists():
        # load precomputed grids
        data = np.load(temp_file)
        T_grid = data["T_grid"]
        rho_grid = data["rho_grid"]
        p_grid = data["p_grid"]
        eta_grid = data["eta_grid"]
        gamma_surface_grid = data["gamma_surface_grid"]
    else:
        rho_grid_mesh, T_grid_mesh = np.meshgrid(mictherm_rho_grid, mictherm_T_grid)
        T_array = T_grid_mesh.flatten()
        rho_array = rho_grid_mesh.flatten()

        # Generate array and calculate mictherm_grid
        mictherm_names, mictherm_units, mictherm_values, InputT, Inputp, Inputrho, Inputx = mictherm_grid(
            mode="userproperties",
            T=T_array,
            rho=rho_array,
            p=None,
            x=np.ones_like(T_array),
            print_output=False,
            base_user_parameters=MIC_THERM_USER_PARAMETERS,
        )
        mictherm_properties = extract_mictherm_properties(
            MIC_THERM_USER_PARAMETERS["properties"],
            mictherm_values,
        )
        T_grid = InputT.reshape(mictherm_T_grid.shape[0], mictherm_rho_grid.shape[0]) / factorTc  # scale back by factorTc
        rho_grid = Inputrho.reshape(mictherm_T_grid.shape[0], mictherm_rho_grid.shape[0]) / factorRho  # scale back by factorRho
        p_grid = mictherm_properties["p"].reshape(mictherm_T_grid.shape[0], mictherm_rho_grid.shape[0]) / factorPc  # scale back by factorPc
        eta_grid = mictherm_properties["eta"].reshape(mictherm_T_grid.shape[0], mictherm_rho_grid.shape[0])
        gamma_surface_grid = mictherm_properties["gamma_surface"].reshape(mictherm_T_grid.shape[0], mictherm_rho_grid.shape[0])

        if debugging == 0:
            np.savez(temp_file, T_grid=T_grid, rho_grid=rho_grid, p_grid=p_grid, eta_grid=eta_grid, gamma_surface_grid=gamma_surface_grid)

    

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

    r = 30
    nx = 200
    ny = 200

    width = 3
    
    a = 9 / 49
    b = 2 / 21
    R = 1.0

    rho_l = 6.764470400
    rho_g = 0.838834226
    Tc = 0.5714285714
    #T = 0.8 * Tc
    T = 0.6 * Tc

    s_rho = [0.0]
    s_e = [1.2]
    s_eta = [1.0]
    s_j = [0.0]
    s_q = [1.0]
    s_v = [1.0]

    
    # Note for Martin: I don't know on which position you want to use eta_grid and gamma_surface_grid, so I just saved them in kwargs for now. You can access them later in the code as needed.
    # for rho,p,T, the grid intepolation has been carried out in MicTherm class, so you can access them directly from the eos object.
    kwargs = {
        "rho_grid": rho_grid, # unit: 'mol/L'
        "p_grid": p_grid, # the p_grid has been saved here in kwargs for later use in the further code, Unit: 'MPa'
        "T_grid": T_grid, # unit: 'K'
        "eta_grid": eta_grid,  # the eta_grid has been saved here in kwargs for later use in the further code, Unit: 'Pa.s'
        "gamma_surface_grid": gamma_surface_grid, # the gamma_surface has been saved here in kwargs for later use in the further code, Unit: 'J/m^2'
    }
    eos = MicTherm(**kwargs)
    # example of how to interpolate properties at a specific state point (rho=12.5, T=280.0)
    properties = interpolate_mictherm_properties(
        kwargs=kwargs,
        rho=12.5,
        T=280.0,
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
        "body_force": [0.0, 0.0],
        "k": [0.01],
        "A": -0.20 * np.ones((1, 1)),
        "M": [M],
        "s_rho": s_rho,
        "s_e": s_e,
        "s_eta": s_eta,
        "s_j": s_j,
        "s_q": s_q,
        "s_v": s_v,
        "kappa": [0.5],
        "precision": precision,
        "io_rate": 10000,
        "compute_MLUPS": False,
        "print_info_rate": 10000,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }

    os.system("rm -rf output*/ *.vtk")
    sim = Droplet2D(**kwargs)
    sim.run(10000)


