"""
Simple 2D thermal BGK-LBM example.

Temperature is fixed hot on the left boundary and cold on the right boundary;
top and bottom are periodic. The velocity is prescribed as zero, so this is
pure heat conduction/diffusion without fluid convection.
"""

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import process_allgather
from termcolor import colored

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.boundary_conditions import ZouHe
from src.lattice import LatticeD2Q9
from src.thermal import BGKSim as ThermalBGK
from src.utils import save_fields_vtk


class ThermalDiffusion2D(ThermalBGK):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0.0, 1.0, self.nx).reshape(self.nx, 1, 1)
        T = T_hot + (T_cold - T_hot) * x
        T = np.broadcast_to(T, (self.nx, self.ny, 1))
        return self.distributed_array_init(T.shape, self.precisionPolicy.compute_dtype, init_val=T)

    def initialize_velocity_field(self):
        u = jnp.zeros((self.nx, self.ny, 2), dtype=self.precisionPolicy.compute_dtype)
        return self.distributed_array_init(u.shape, self.precisionPolicy.compute_dtype, init_val=u)

    def assign_fields_sharded(self):
        T0 = self.initialize_macroscopic_fields()
        u0 = self.initialize_velocity_field()
        return self.initialize_populations(T0, u0)

    def set_thermal_boundary_conditions(self):
        left = self.boundingBoxIndices["left"]
        right = self.boundingBoxIndices["right"]

        T_left = T_hot * np.ones((left.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        T_right = T_cold * np.ones((right.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)

        self.thermal_BCs.append(ZouHe(tuple(left.T), self.gridInfo, self.precisionPolicy, "pressure", T_left))
        self.thermal_BCs.append(ZouHe(tuple(right.T), self.gridInfo, self.precisionPolicy, "pressure", T_right))

    def run(self, t_max):
        g = self.assign_fields_sharded()
        u = self.initialize_velocity_field()

        for timestep in range(t_max + 1):
            io_flag = self.ioRate > 0 and (timestep % self.ioRate == 0 or timestep == t_max)
            print_iter_flag = self.printInfoRate > 0 and timestep % self.printInfoRate == 0

            if timestep > 0:
                g, _ = self.step(g, u, timestep)

            if print_iter_flag:
                print(colored("Timestep ", "blue") + colored(f"{timestep}", "green") + colored(" completed", "blue"))

            if io_flag:
                T = process_allgather(self.compute_temperature(g))
                self.output_data(timestep=timestep, T=T)

        return g

    def output_data(self, **kwargs):
        T = np.asarray(kwargs["T"])

        if T.ndim == 4 and T.shape[0] == 1:
            T = T[0]

        dT_dx, dT_dy = np.gradient(T[..., 0])
        qx = -thermal_diffusivity * dT_dx
        qy = -thermal_diffusivity * dT_dy

        fields = {
            "T": T[..., 0],
            "T_norm": (T[..., 0] - T_cold) / (T_hot - T_cold),
            "qx": qx,
            "qy": qy,
        }
        save_fields_vtk(kwargs["timestep"], fields, "output_thermal_diffusion", "data")


if __name__ == "__main__":
    precision = "f32/f32"
    nx = 128
    ny = 64

    T_hot = 1.0
    T_cold = 0.5

    # Heat transfer is controlled by thermal_diffusivity through:
    # thermal_diffusivity = (1 / thermal_omega - 0.5) / 3
    thermal_diffusivity = 0.05
    thermal_omega = 1.0 / (3.0 * thermal_diffusivity + 0.5)

    lattice = LatticeD2Q9(precision)

    common_kwargs = {
        "lattice": lattice,
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "precision": precision,
        "io_rate": 100,
        "print_info_rate": 100,
        "checkpoint_rate": -1,
    }

    sim = ThermalDiffusion2D(
        **common_kwargs,
        omega=thermal_omega,
    )

    os.system("if exist output_thermal_diffusion rmdir /s /q output_thermal_diffusion")
    sim.run(3000)
