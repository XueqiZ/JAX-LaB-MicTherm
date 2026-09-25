"""Single-component 2D capillary rise using the MicTherm PeTS EOS. combines the fingering and capillary rise test cases. 
The capillary rise is driven by a body force and the contact angle is set by the wall wetting parameters theta, phi and delta_rho. 
The simulation results are saved in VTK format for visualization in Paraview."""

import os
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lattice import LatticeD2Q9
from src.boundary_conditions import BounceBack
from MicLBM_src.eos import MicTherm
from MicLBM_src.Mic_multiphase import MultiphaseMRTTvar
from MicLBM_src.Mic_utils import save_fields_vtk



# Estimate surface tension
class Droplet2D(MultiphaseMRTTvar):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)
        x = x.T
        y = y.T
        dist = np.sqrt((x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2)
        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree = []
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
        # Use one offset so all four sampling points have the same distance
        # from the droplet center. The boundary margin prevents invalid indices.
        boundary_margin = 10
        offset = min(40, self.nx // 2 - boundary_margin, self.ny // 2 - boundary_margin)
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
        if timestep == 60000:
            file.write(f"{1 / r},{pressure_difference}\n")
        save_fields_vtk(timestep, fields, f"output_{r}", "data")


# Estimate contact angle, later when we have the correct parameters for the substance, we can carried out once to estaimate 
# the contact angle and then use the same parameters for the capillary rise simulation. It takes more then 3h on my pc to calculate the contact angle for one set of parameters. 
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)
        x = x.T
        y = y.T
        dist = np.sqrt((x - self.nx / 2) ** 2 + (y + disp) ** 2)
        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree = []
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        # Define wall boundary indices: combine top and bottom bounding-box indices.
        # Convert to a tuple of coordinate arrays (required by the BounceBack BC).
        # Apply full-way bounce-back on these wall nodes and pass the
        # wetting parameters (theta, phi, delta_rho) evaluated at the wall
        # locations to model fluid-solid interactions (contact angle / slip).
        walls = np.concatenate((self.boundingBoxIndices["top"], self.boundingBoxIndices["bottom"]))
        walls = tuple(walls.T)
        self.BCs[0].append(BounceBack(walls, self.gridInfo, self.precisionPolicy, theta[walls], phi[walls], delta_rho[walls]))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        p = np.array(kwargs["p"][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {"p": p[..., 0], "rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1]}
        save_fields_vtk(timestep, fields, f"output_{disp}", "data")


class CapillaryRise2D(MultiphaseMRTTvar):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        rho_profile = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (x - L) / width)
        rho = rho_g * np.ones((self.nx, self.ny, 1))
        rho[:, :, 0] = rho_profile.reshape((self.nx, 1))

        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree = [rho]

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u]

        return rho_tree, u_tree

    def set_boundary_conditions(self):
        top_wall = np.array(
            [[x, y] for x in range(150, 451) for y in range(29)],
            dtype=np.int32,
        )
        bottom_wall = np.array(
            [[x, y] for x in range(150, 451) for y in range(self.ny - 28, self.ny)],
            dtype=np.int32,
        )
        walls = np.concatenate((top_wall, bottom_wall))
        walls = tuple(walls.T)
        self.BCs[0].append(BounceBack(walls, self.gridInfo, self.precisionPolicy, theta[walls], phi[walls], delta_rho[walls]))

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        p = np.array(kwargs["p"][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        T_field = np.array(self.T_field)
        fields = {
            "flag": self.solid_mask_streamed[0][..., 0],
            "p": p[..., 0],
            "rho": rho[..., 0],
            "T": T_field[..., 0],
            "velocity": u,
            "velocity_magnitude": np.linalg.norm(u, axis=-1),
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        save_fields_vtk(timestep, fields, "output", "data")
        ind_mid = np.argmin(p[150:451, self.ny // 2, 0])
        ind_side = np.argmin(p[150:451, self.ny - 30, 0])
        meniscus_position = ind_mid
        meniscus_height = ind_side - ind_mid
        file.write(f"{timestep},{meniscus_height},{meniscus_position}\n")


MIC_THERM_USER_PARAMETERS = {
    "units": "REDUCED",
    "Output": "no",
    "Debug": "no",
    "stability": "no",
    "N_components": 1,
    "Substance_ID1": 0,
    "PotModel_1": "LJ.pm",
    "EOS": "PeTS",
    "epsilon_1": 1,
    "sigma_1": 1,
    "chainlength": 1,
    "molar_mass_1": 1,
    "eta_param": "0; -0.075; 0.756; 0.4; 0.113",
    "DGT_kappa_1": 2.7334,
    "transPropMode": "entropyScaling",
    "properties": "p, eta, gamma_surface",
}


def load_mictherm_pets_eos():
    """Load a precomputed PeTS EOS table and its VLE curve."""
    temp_dir = PROJECT_ROOT / "temp"
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
        tables = {
            name: data[name]
            for name in (
                "rho_grid",
                "p_grid",
                "T_grid",
                "eta_grid",
                "gamma_surface_grid",
            )
        }
        vle_temperature = data["VLE_T"]
        vle_liquid_density = data["VLE_rho_l"]
        vle_vapor_density = data["VLE_rho_g"]

    return (
        MicTherm(**tables),
        Tc,
        vle_temperature,
        vle_liquid_density,
        vle_vapor_density,
    )


if __name__ == "__main__":
    nx = 600
    ny = 100
    width = 6  # Diffuse liquid-vapor interface thickness used by the PeTS case

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

    s_rho = [0.0]
    s_e = [1.2]
    s_eta = [1.0]
    s_j = [0.0]
    s_q = [1.0]
    s_v = [1.0]

    eos, Tc, micthermVLE_T, micthermVLE_rho_l, micthermVLE_rho_g = load_mictherm_pets_eos()

    try:
        Tr = float(input("Define reduced base temperature Tr (e.g. 0.85): "))
    except Exception:
        Tr = 0.85
    T = Tr * Tc

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

    temperature_initialization = input(
        "Select temperature initialization ('constant' or 'gradient'): "
    ).strip().lower()
    if temperature_initialization not in ("constant", "gradient"):
        temperature_initialization = "constant"

    if temperature_initialization == "constant":
        T_field = np.full((nx, ny), T, dtype=float)
    else:
        try:
            t_left_mul = float(input("Define left-boundary temperature multiplier (e.g. 1.1): "))
        except Exception:
            t_left_mul = 1.1
        try:
            t_right_mul = float(input("Define right-boundary temperature multiplier (e.g. 0.9): "))
        except Exception:
            t_right_mul = 0.9
        x_profile = np.linspace(t_left_mul * T, t_right_mul * T, nx, dtype=float)
        T_field = np.repeat(x_profile[:, np.newaxis], ny, axis=1)

    T_field = T_field.reshape((nx, ny, 1))

    precision = "f32/f32"

    # Estimate surface tension: Prüft, welche Oberflächenspannung das gewählte Parameterset (k, kappa, A, EOS usw.) tatsächlich 
    # numerisch erzeugt. Dazu wird das Laplace-Gesetz \(\Delta p=\sigma/R\) verwendet. Diese \(\sigma\) brauchst du, 
    # um die Kapillarsteigung mit der Lucas–Washburn-Theorie zu vergleichen.

    # os.system("rm -rf output* surface_tension.txt")
    # file = open("surface_tension.txt", "w")
    # file.write("Inverse Radius,Pressure Difference\n")
    # file.write("0,0\n")
    # for r in [20, 25, 30, 35]:
    #     kwargs = {
    #         "n_components": 1,
    #         "lattice": LatticeD2Q9(precision),
    #         "nx": nx,
    #         "ny": ny,
    #         "nz": 0,
    #         "body_force": [0.0, 0.0],
    #         "g_kkprime": -1 * np.ones((1, 1)),
    #         "k": [0.15],
    #         "A": -0.115 * np.ones((1, 1)),
    #         "M": [M],
    #         "s_rho": s_rho,
    #         "s_e": s_e,
    #         "s_eta": s_eta,
    #         "s_j": s_j,
    #         "s_q": s_q,
    #         "s_v": s_v,
    #         "kappa": [0.5],
    #         "precision": precision,
    #         "EOS": eos,
    #         "io_rate": 60000,
    #         "print_info_rate": 60000,
    #         "checkpoint_rate": -1,
    #         "checkpoint_dir": os.path.abspath("./checkpoints_"),
    #         "restore_checkpoint": False,
    #     }
    #     sim = Droplet2D(**kwargs)
    #     sim.run(60000)
    # file.close()

    # entscheide dich für die Wandbenetzung: theta, phi und delta_rho. Diese Parameter bestimmen den Kontaktwinkel der Flüssigkeit an der Wand.
    theta = 19.1 * (np.pi / 180) * np.ones((nx, ny, 1))
    phi = 1.1 * np.ones((nx, ny, 1))
    delta_rho = np.zeros((nx, ny, 1))

    # Estimate contact angle: Prüft, welchen realen Kontaktwinkel die Wandparameter theta, phi und delta_rho tatsächlich erzeugen.
    # In einer Pseudopotential-LBM kann der gemessene Gleichgewichtswinkel vom eingegebenen theta = 19.1° abweichen. 
    # Die Tropfentests zeigen, ob die Wandbenetzung korrekt eingestellt ist.
    # os.system("rm -rf output* *.vtk")
    # r = 50
    # Disp = [0, 10, 20, 30]
    # for disp in Disp:
    #     kwargs = {
    #         "n_components": 1,
    #         "lattice": LatticeD2Q9(precision),
    #         "nx": nx,
    #         "ny": ny,
    #         "nz": 0,
    #         "body_force": [0.0, 0.0],
    #         "g_kkprime": -1 * np.ones((1, 1)),
    #         "k": [0.15],
    #         "A": -0.115 * np.ones((1, 1)),
    #         "M": [M],
    #         "s_rho": s_rho,
    #         "s_e": s_e,
    #         "s_eta": s_eta,
    #         "s_j": s_j,
    #         "s_q": s_q,
    #         "s_v": s_v,
    #         "kappa": [0.5],
    #         "precision": precision,
    #         "io_rate": 40000,
    #         "EOS": eos,
    #         "print_info_rate": 40000,
    #         "checkpoint_rate": -1,
    #         "checkpoint_dir": os.path.abspath("./checkpoints_"),
    #         "restore_checkpoint": False,
    #     }
    #     sim = DropletOnSurface2D(**kwargs)
    #     sim.run(400000)
    
    # entscheide dich für die Wandbenetzung: theta, phi und delta_rho. Diese Parameter bestimmen den Kontaktwinkel der Flüssigkeit an der Wand.    
    L = 150
    theta = 19.1 * (np.pi / 180) * np.ones((nx, ny, 1))
    phi = 1.1 * np.ones((nx, ny, 1))
    delta_rho = np.zeros((nx, ny, 1))

    file = open("lucas_washburn.txt", "w")
    file.write("Time,Menicus Height,Position\n")
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD2Q9(precision),
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "body_force": [1.5 * 1e-5, 0.0],
        "g_kkprime": -1 * np.ones((1, 1)),
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
        "EOS": eos,
        "T_field": T_field,
        "precision": precision,
        "io_rate": 50,
        "print_info_rate": 50,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }
    sim = CapillaryRise2D(**kwargs)
    steps = input("How many steps should be run after starting the simulation? [3000]: ")
    steps = int(steps) if steps.strip() else 3000
    sim.run(steps)
    file.close()
