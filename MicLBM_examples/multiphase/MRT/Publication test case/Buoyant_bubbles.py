"""MicTherm VdW heterogeneous boiling from three wall nucleation sites.

The numerical setup follows the stable
``droplet_2d_2D_Mic_VdW_Trinangle_Profil`` parameters. The domain initially
contains liquid at uniform temperature. After an isothermal relaxation period,
a hot-bottom/cold-top profile and three hydrophobic hot spots are active.
"""

import os
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MicLBM_src.Mic_multiphase import MultiphaseMRTTvar
from MicLBM_src.Mic_utils import save_fields_vtk
from MicLBM_src.eos import MicTherm
from MicLBM_src.run_mictherm import (
    extract_mictherm_properties,
    mictherm_grid,
    run_mictherm_func,
)
from src.boundary_conditions import BounceBack
from src.lattice import LatticeD2Q9


class HeterogeneousBoiling2D(MultiphaseMRTTvar):
    """LBM-Fall für heterogenes Blasensieden an drei Keimstellen."""

    def initialize_macroscopic_fields(self):
        """Erzeuge das anfängliche Dichte- und Geschwindigkeitsfeld."""
        x, y = np.meshgrid(
            np.arange(self.nx), np.arange(self.ny), indexing="ij"
        )
        rho = np.full((self.nx, self.ny), rho_l, dtype=float)
        # Only solid nucleation-site nodes receive vapor-like pseudo-density.
        # All fluid nodes start as liquid; no pre-existing bubble is imposed.
        rho[nucleation_site_mask] = rho_g

        rho = rho[..., None]
        rho = self.distributed_array_init(
            (self.nx, self.ny, 1),
            self.precisionPolicy.compute_dtype,
            init_val=rho,
        )
        rho = self.precisionPolicy.cast_to_output(rho)

        velocity = np.zeros((self.nx, self.ny, 2))
        velocity = self.distributed_array_init(
            (self.nx, self.ny, 2),
            self.precisionPolicy.compute_dtype,
            init_val=velocity,
        )
        velocity = self.precisionPolicy.cast_to_output(velocity)
        return [rho], [velocity]

    def set_boundary_conditions(self):
        """Setze No-Slip-Wände oben/unten; links/rechts bleiben periodisch."""
        top = tuple(self.boundingBoxIndices["top"].T)
        bottom = tuple(self.boundingBoxIndices["bottom"].T)
        self.BCs[0].append(
            BounceBack(top, self.gridInfo, self.precisionPolicy)
        )
        self.BCs[0].append(
            BounceBack(
                bottom,
                self.gridInfo,
                self.precisionPolicy,
                theta=wall_theta[bottom],
                phi=wall_phi[bottom],
                delta_rho=wall_delta_rho[bottom],
            )
        )

    def output_data(self, **kwargs):
        """Schreibe Zustandsfelder für ParaView und protokolliere Extremwerte."""
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        pressure = np.array(kwargs["p"][0, ...])
        velocity = np.array(kwargs["u_tree"][0][0, ...])
        temperature = np.array(self.T_field)
        timestep = kwargs["timestep"]

        fields = {
            "rho": rho[..., 0],
            "p": pressure[..., 0],
            "T": temperature[..., 0],
            "velocity": velocity,
            "ux": velocity[..., 0],
            "uy": velocity[..., 1],
            "nucleation_sites": nucleation_site_mask.astype(float),
        }
        save_fields_vtk(timestep, fields, "output_heterogeneous_boiling", "data")

        speed = np.linalg.norm(velocity, axis=-1)
        print(
            f"rho min/max={rho.min():.8g}/{rho.max():.8g}; "
            f"max velocity={speed.max():.8g}"
        )


def select_cached_mictherm_eos():
    """Wähle einen vollständigen NPZ-Cache und erzeuge daraus das EOS-Objekt."""

    # 1. Alle Felder festlegen, die ein verwendbarer Cache enthalten muss.
    temp_dir = PROJECT_ROOT / "temp"
    required = {
        "Tc", "VLE_T", "VLE_rho_l", "VLE_rho_g", "T_grid", "rho_grid",
        "p_grid", "eta_grid", "gamma_surface_grid",
    }
    candidates = []

    # 2. Nur vollständige Archive in die Auswahlliste aufnehmen.
    for path in sorted(temp_dir.glob("temp_mictherm_grids*.npz")):
        with np.load(path) as data:
            if required.issubset(data.files):
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No complete MicTherm cache with critical and VLE data found in {temp_dir}."
        )

    # 3. Verfügbare Archive anzeigen; PeTS ist die bevorzugte Standardwahl. -> stabil
    print("Compatible MicTherm cache files:")
    for index, path in enumerate(candidates, start=1):
        print(f"  {index}) {path.name}")
    default_choice = next(
        (
            index
            for index, path in enumerate(candidates, start=1)
            if path.name == "temp_mictherm_grids_PeTS.npz"
        ),
        1,
    )
    try:
        # 4. Benutzereingabe prüfen und bei leerer Eingabe den Standard nutzen.
        choice_text = input(
            f"Select file [1-{len(candidates)}] [{default_choice}]: "
        ).strip()
        choice = int(choice_text) if choice_text else default_choice
        if not 1 <= choice <= len(candidates):
            raise ValueError
    except ValueError:
        print("Invalid selection; using the first compatible file.")
        choice = 1
    temp_file = candidates[choice - 1]

    # 5. EOS-Gitter, kritische Temperatur und Koexistenzkurven laden.
    with np.load(temp_file) as data:
        tables = {
            name: data[name]
            for name in (
                "T_grid", "rho_grid", "p_grid", "eta_grid",
                "gamma_surface_grid",
            )
        }
        critical_temperature = data["Tc"].item()
        vle_temperature = data["VLE_T"]
        vle_liquid_density = data["VLE_rho_l"]
        vle_vapor_density = data["VLE_rho_g"]

        # 6. VdW-VLE-Daten liegen noch in MicTherm-Einheiten vor, während die
        #    zugehörigen Gitter bereits reduzierte LBM-Einheiten verwenden.
        if "factorTc" in data.files:
            vle_temperature = vle_temperature / data["factorTc"].item()

        if "factorRho" in data.files:
            density_factor = data["factorRho"].item()
            vle_liquid_density = vle_liquid_density / density_factor
            vle_vapor_density = vle_vapor_density / density_factor

    # 7. Fertiges Tabellen-EOS und die Daten für die VLE-Interpolation liefern.
    return (
        MicTherm(**tables),
        critical_temperature,
        vle_temperature,
        vle_liquid_density,
        vle_vapor_density,
        temp_file.name,
    )


def mrt_matrix():
    """Erzeuge die D2Q9-Transformationsmatrix für die MRT-Kollision."""

    # Diskrete Gittergeschwindigkeiten und deren Beträge.
    lattice_velocity = LatticeD2Q9().c.T
    norm = np.linalg.norm(lattice_velocity, axis=1)
    # Jede Zeile beschreibt ein Dichte-, Impuls- oder Spannungsmoment.
    matrix = np.zeros((9, 9))
    matrix[0, :] = norm**0
    matrix[1, :] = -4 * norm**0 + 3 * norm**2
    matrix[2, :] = 4 * norm**0 - 10.5 * norm**2 + 4.5 * norm**4
    matrix[3, :] = lattice_velocity[:, 0]
    matrix[4, :] = (-5 * norm**0 + 3 * norm**2) * lattice_velocity[:, 0]
    matrix[5, :] = lattice_velocity[:, 1]
    matrix[6, :] = (-5 * norm**0 + 3 * norm**2) * lattice_velocity[:, 1]
    matrix[7, :] = lattice_velocity[:, 0] ** 2 - lattice_velocity[:, 1] ** 2
    matrix[8, :] = lattice_velocity[:, 0] * lattice_velocity[:, 1]
    return matrix


if __name__ == "__main__":
    # --- Simulationsgebiet -------------------------------------------------
    # Anzahl der LBM-Zellen in horizontaler und vertikaler Richtung.
    nx = 200
    ny = 200

    # --- Stoffdaten und Koexistenzdichten ---------------------------------
    # EOS-Tabelle und VLE-Daten aus dem ausgewählten NPZ-Archiv übernehmen.
    (
        eos,
        Tc,
        micthermVLE_T,
        micthermVLE_rho_l,
        micthermVLE_rho_g,
        selected_cache,
    ) = select_cached_mictherm_eos()

    # Reduzierte Arbeitstemperatur einlesen und in die Einheit des Caches
    # umrechnen.
    try:
        Tr_text = input("Define reduced base temperature Tr [0.80]: ").strip()
        Tr = float(Tr_text) if Tr_text else 0.80
    except ValueError:
        print("Invalid temperature; using Tr=0.80.")
        Tr = 0.80
    T_reference = Tr * Tc

    # Extrapolation außerhalb der gespeicherten Koexistenzkurve verhindern.
    vle_T_min = float(np.min(micthermVLE_T))
    vle_T_max = float(np.max(micthermVLE_T))
    if not vle_T_min <= T_reference <= vle_T_max:
        raise ValueError(
            f"T={T_reference} is outside the selected cache VLE range "
            f"[{vle_T_min}, {vle_T_max}]."
        )

    # Flüssigkeits- und Dampfdichte als Funktion der Temperatur interpolieren.
    interp_rho_l = interp1d(
        micthermVLE_T,
        micthermVLE_rho_l,
        kind="cubic",
    )
    interp_rho_g = interp1d(
        micthermVLE_T,
        micthermVLE_rho_g,
        kind="cubic",
    )
    rho_l = float(interp_rho_l(T_reference))
    rho_g = float(interp_rho_g(T_reference))
    print(
        f"Using {selected_cache}: Tc={Tc:.8g}, T={T_reference:.8g}, "
        f"rho_l={rho_l:.8g}, rho_g={rho_g:.8g}"
    )

    # --- Keimstellen am unteren Rand --------------------------------------
    # Drei thermische und hydrophobe Keimstellen erzeugen.
    nucleation_site_centers = np.array([0.25, 0.50, 0.75]) * nx
    wetting_half_width = 3.0
    hotspot_sigma_x = 6.0
    hotspot_sigma_y = 4.0

    x_coordinate = np.arange(nx, dtype=float)[:, None]
    y_coordinate = np.arange(ny, dtype=float)[None, :]
    nucleation_site_mask = np.zeros((nx, ny), dtype=bool)
    thermal_hotspots = np.zeros((nx, ny), dtype=float)
    for center_x in nucleation_site_centers:
        dx = np.abs(x_coordinate - center_x)
        dx = np.minimum(dx, nx - dx)
        nucleation_site_mask[:, 0] |= dx[:, 0] <= wetting_half_width
        thermal_hotspots += np.exp(
            -0.5 * (dx / hotspot_sigma_x) ** 2
            -0.5 * (y_coordinate / hotspot_sigma_y) ** 2
        )

    # --- Benetzungseigenschaften der Wand ---------------------------------
    # Die untere Wand ist neutral; nur die Keimstellen sind hydrophob.
    wall_theta = 0.5 * np.pi * np.ones((nx, ny, 1))
    wall_phi = np.ones((nx, ny, 1))
    wall_delta_rho = np.zeros((nx, ny, 1))
    wall_theta[nucleation_site_mask, 0] = np.deg2rad(150.0)
    wall_delta_rho[nucleation_site_mask, 0] = (
        0.1 * (rho_l - rho_g)
    )

    # --- Temperaturfelder -------------------------------------------------
    # Hintergrundgradient plus glatte lokale Wärmequellen. Clipping verhindert,
    # dass das Temperaturfeld den vorgesehenen Tabellenbereich stark verlässt.
    T_bottom = 1.02 * T_reference
    T_top = 0.90 * T_reference
    hotspot_delta_T = 0.08 * T_reference
    vertical_coordinate = np.linspace(0.0, 1.0, ny)
    vertical_temperature = (
        T_bottom + (T_top - T_bottom) * vertical_coordinate
    )
    background_gradient_T_field = np.tile(
        vertical_temperature, (nx, 1)
    )
    hotspot_T_field = (
        background_gradient_T_field
        + hotspot_delta_T * thermal_hotspots
    )
    hotspot_T_field = np.clip(
        hotspot_T_field,
        0.90 * T_reference,
        1.10 * T_reference,
    )[..., None]
    background_gradient_T_field = background_gradient_T_field[..., None]
    uniform_T_field = np.full(
        (nx, ny, 1), T_reference, dtype=float
    )

    # --- Zeitsteuerung der thermischen Phasen -----------------------------
    # Zuerst isotherme Relaxation, danach Hotspots und anschließend optional
    # nur noch der Hintergrundgradient.
    temperature_switch_timestep = 500
    disable_hotspots_after_growth = True
    hotspot_off_timestep = 16000
    # Optional wird nach dem Blasenwachstum auch die hydrophobe Wandbedingung
    # entfernt. phi > 1 macht die ehemaligen Keimstellen flüssigkeitsaffin.
    disable_hydrophobic_sites_after_growth = True
    post_growth_wall_phi = 1.05
    final_timestep = 25000
    if disable_hotspots_after_growth and not (
        temperature_switch_timestep
        < hotspot_off_timestep
        < final_timestep
    ):
        raise ValueError(
            "hotspot_off_timestep must lie strictly between "
            "temperature_switch_timestep and final_timestep"
        )

    # --- Äußere Kraft und LBM-Parameter -----------------------------------
    # Schwerkraft wirkt nach unten; für einen reinen Temperaturtest auf null.
    gravity_acceleration = 1.0e-7
    # gravity_acceleration = 0
    precision = "f32/f32"
    # Zentrale Konfiguration des Mehrphasen-MRT-Solvers.
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD2Q9(precision),
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "g_kkprime": -1.0 * np.ones((1, 1)),
        "EOS": eos,
        # Die erste Solverphase verwendet ein homogenes Temperaturfeld.
        "T_field": uniform_T_field,
        "body_force": [0.0, -gravity_acceleration],
        "k": [0.16],
        "A": -0.032 * np.ones((1, 1)),
        "M": [mrt_matrix()],
        "s_rho": [0.0],
        "s_e": [1.2],
        "s_eta": [1.0],
        "s_j": [0.0],
        "s_q": [1.0],
        "s_v": [1.0],
        "kappa": [0.4],
        "precision": precision,
        "io_rate": 100,
        "compute_MLUPS": False,
        "print_info_rate": 100,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_thermal_droplet"),
        "restore_checkpoint": False,
    }

    # --- Phase 1: isotherme Relaxation ------------------------------------
    # Das Dichtefeld darf sich zunächst ohne thermischen Sprung stabilisieren.
    print(
        f"Isothermal relaxation through timestep "
        f"{temperature_switch_timestep}."
    )
    simulation = HeterogeneousBoiling2D(**kwargs)
    f_tree = simulation.run(temperature_switch_timestep)

    # --- Phase 2: Wachstum an den beheizten Keimstellen -------------------
    # Verteilungen aus Phase 1 werden ohne Neuinitialisierung weitergegeben.
    print(
        f"Applying hot-bottom/cold-top temperature gradient at timestep "
        f"{temperature_switch_timestep + 1}."
    )
    gradient_kwargs = dict(kwargs)
    gradient_kwargs.update(
        {
            "T_field": hotspot_T_field,
            "checkpoint_rate": -1,
            "restore_checkpoint": False,
        }
    )
    simulation = HeterogeneousBoiling2D(**gradient_kwargs)
    # --- Phase 3: Hotspots optional abschalten ----------------------------
    if disable_hotspots_after_growth:
        print(
            f"Growing vapor nuclei with local hot spots through timestep "
            f"{hotspot_off_timestep}."
        )
        f_tree = simulation.run(
            hotspot_off_timestep,
            initial_f_tree=f_tree,
            start_step=temperature_switch_timestep + 1,
        )

        print(
            f"Disabling local hot spots at timestep "
            f"{hotspot_off_timestep + 1}; keeping only the background "
            f"temperature gradient."
        )
        background_kwargs = dict(gradient_kwargs)
        background_kwargs["T_field"] = background_gradient_T_field

        # Die Änderung der Benetzung ist getrennt schaltbar.
        if disable_hydrophobic_sites_after_growth:
            print(
                "Switching former nucleation sites from hydrophobic to "
                f"liquid-wetting (phi={post_growth_wall_phi})."
            )
            wall_theta[nucleation_site_mask, 0] = 0.5 * np.pi
            wall_delta_rho[nucleation_site_mask, 0] = 0.0
            wall_phi[nucleation_site_mask, 0] = post_growth_wall_phi

        simulation = HeterogeneousBoiling2D(**background_kwargs)
        simulation.run(
            final_timestep,
            initial_f_tree=f_tree,
            start_step=hotspot_off_timestep + 1,
        )
    else:
        print("Local hot spots remain active through the final timestep.")
        simulation.run(
            final_timestep,
            initial_f_tree=f_tree,
            start_step=temperature_switch_timestep + 1,
        )
