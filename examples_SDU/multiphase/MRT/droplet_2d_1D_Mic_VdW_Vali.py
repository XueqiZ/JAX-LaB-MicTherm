"""
Single component 2D droplet example where liquid droplet is suspended in its vapor. The density of each region is computed using Maxwell's Construction. The density profile
is initialized with smooth profile with specified interface width. Boundary conditions are periodic everywhere. Useful for tuning the various coefficients.

The collision matrix is based on:
1. McCracken, M. E. & Abraham, J. Multiple-relaxation-time lattice-Boltzmann model for multiphase flow. Phys. Rev. E 71, 036701 (2005).
"""

import os
import sys
import csv
from pathlib import Path
from jax import config
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_mictherm import run_mictherm_func, mictherm_grid
from src.lattice import LatticeD2Q9
from src.eos import MicTherm
from src.utils import *
from src.multiphase import MultiphaseMRT
from validate_thermodynamic_consistency_vdw import read_thermodynamic_consistency

# config.update("jax_default_matmul_precision", "float32")


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

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        p = np.array(kwargs["p"][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {"p": p[..., 0], "rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1]}
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
        save_fields_vtk(timestep, fields, output_dir, "data")

        if timestep == validation_run_steps:
            with open(validation_results_path, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        current_Tr,
                        current_k,
                        current_A,
                        float(spurious_currents),
                        float(rho_g),
                        float(rho_g_pred),
                        float(rho_l),
                        float(rho_l_pred),
                    ]
                )

        


if __name__ == "__main__":
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
    )

    # reference: publication 
    Tc = 4/7
    rhoc = 7/2
    pc = (9/49) / (27 * (2/21) ** 2)

    factorTc = mictherm_values[0,0]/Tc
    factorRho = mictherm_values[0,1]/rhoc
    factorPc = mictherm_values[0,2]/pc

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


    s_rho = [0.0]
    s_e = [1.2]
    s_eta = [1.0]
    s_j = [0.0]
    s_q = [1.0]
    s_v = [1.0]

    general_information, pd_values, rms_error = read_thermodynamic_consistency()
    print(f"Loaded {len(pd_values)} validation cases from Thermodynamic Consistency - VdW.csv")
    print(f"Reference RMS Error from CSV: {rms_error}")

    validation_output_root = os.path.abspath("output_validation")
    os.makedirs(validation_output_root, exist_ok=True)
    validation_results_path = os.path.join(validation_output_root, "validation_results.csv")
    validation_headers = [
        "Tr",
        "k",
        "A",
        "Spurious Currents (Max)",
        "rho_g",
        "rho_g_pred",
        "rho_l",
        "rho_l_pred",
    ]
    with open(validation_results_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(validation_headers)

    validation_run_steps = 10000
    precision = "f32/f32"
    for case_index, validation_case in pd_values.iterrows():
        Tr = float(validation_case["Tr"])
        k_value = float(validation_case["k"])
        A_value = float(validation_case["A"])
        T = Tr * Tc
        current_Tr = Tr
        current_k = k_value
        current_A = A_value

        print(f"\nStarting validation case {case_index}: Tr={Tr}, k={k_value}, A={A_value}")

        # calculate VLE properties using MicTherm API for Tiso
        mictherm_names, mictherm_units, mictherm_values = run_mictherm_func(
            mode="vle_iso",
            t_iso=T * factorTc,  # scale Tiso by factorTc to be consistent with critical point properties
            print_output=False,
        )
        mictherm_rho_l = mictherm_values[0, 1]
        mictherm_rho_g = mictherm_values[0, 2]
        mictherm_T = T * factorTc  # scale T by factorTc to be consistent with critical point properties
        rho_l = mictherm_rho_l / factorRho  # scale back by factorRho
        rho_g = mictherm_rho_g / factorRho  # scale back by factorRho
        rho_array = np.linspace(mictherm_rho_g * 0.7, mictherm_rho_l * 1.3, 100)
        T_array = np.full_like(rho_array, mictherm_T)

        mictherm_names, mictherm_units, mictherm_values, InputT, Inputp, Inputrho, Inputx = mictherm_grid(
            mode="userproperties",
            T=T_array,
            rho=rho_array,
            p=None,
            x=np.ones_like(T_array),
            print_output=False,
        )

        p_grid = mictherm_values[:, 1] / factorPc
        rho_grid = Inputrho / factorRho

        kwargs = {"rho_grid": rho_grid, "p_grid": p_grid, "T": T}
        eos = MicTherm(**kwargs)

        case_name = f"Tr_{Tr:.3f}".replace(".", "_")
        output_dir = os.path.abspath(os.path.join(validation_output_root, case_name))
        os.makedirs(output_dir, exist_ok=True)

        kwargs = {
            "n_components": 1,
            "lattice": LatticeD2Q9(precision),
            "nx": nx,
            "ny": ny,
            "nz": 0,
            "g_kkprime": -1.0 * np.ones((1, 1)),
            "EOS": eos,
            "body_force": [0.0, 0.0],
            "k": [k_value],
            "A": A_value * np.ones((1, 1)),
            "M": [M],
            "s_rho": s_rho,
            "s_e": s_e,
            "s_eta": s_eta,
            "s_j": s_j,
            "s_q": s_q,
            "s_v": s_v,
            "kappa": [1.0],
            "precision": precision,
            "io_rate": 2000,
            "compute_MLUPS": False,
            "print_info_rate": 2000,
            "checkpoint_rate": -1,
            "checkpoint_dir": os.path.abspath(os.path.join("checkpoints_validation", case_name)),
            "restore_checkpoint": False,
        }

        sim = Droplet2D(**kwargs)
        sim.run(validation_run_steps)
