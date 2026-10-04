"""Calculate and plot isothermal VLE curves for the HFO-1234yf/CO2 mixture."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.tri import LinearTriInterpolator, Triangulation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MicLBM_src.run_mictherm import run_mictherm_func


MIC_THERM_USER_PARAMETERS = {
    "units": "SI",
    "Output": "no",
    "Debug": "no",
    "stability": "no",
    "N_components": 2,
    "Substance_ID1": 103,  # CO2
    "PotModel_1": "LJ.pm",
    "Substance_ID2": 0,  # HFO-1234yf
    "PotModel_2": "LJ.pm",
    "EOS": "PC_SAFT",
    "epsilon_2": 179.953,
    "sigma_2": 3.631,
    "chainlength_2": 2.358,
    "molar_mass_2": 114.04,
    "Polar_2": "GrossVrabec",
    "Dipolemoment_2": 2.011,
    "Dipolemoment_xp_2": 0.33,
}


def ask_float(prompt, default):
    value = input(f"{prompt} [{default}]: ").strip()
    return float(value) if value else float(default)


def inclusive_temperature_range(T_low, T_high, T_step):
    """Return a range that includes T_high when it lies on the requested grid."""
    count = int(np.floor((T_high - T_low) / T_step + 1.0e-12)) + 1
    temperatures = T_low + T_step * np.arange(count, dtype=float)
    if temperatures[-1] < T_high and np.isclose(
        temperatures[-1] + T_step, T_high
    ):
        temperatures = np.append(temperatures, T_high)
    return temperatures


RESULT_NAMES = (
    "T",
    "VLE_x_l",
    "p_VLE",
    "VLE_y_v",
    "VLE_rho_l",
    "VLE_rho_v",
)


def calculate_vle_scan():
    T_low = ask_float("Lowest VLE temperature T_low", 270.0)
    T_high = ask_float("Highest VLE temperature T_high", 360.0)
    T_step = ask_float("VLE temperature step T_step", 10.0)

    if T_low > T_high:
        raise ValueError("T_low must not be greater than T_high")
    if T_step <= 0.0:
        raise ValueError("T_step must be positive")

    temperatures = inclusive_temperature_range(T_low, T_high, T_step)
    results = {name: [] for name in RESULT_NAMES}

    for index, T_iso in enumerate(temperatures, start=1):
        print(
            f"Calculating isothermal VLE at T={T_iso:g} "
            f"({index}/{temperatures.size})..."
        )
        _, _, values = run_mictherm_func(
            mode="vle_iso",
            T=None,
            rho=None,
            p=None,
            x=1.0,
            t_iso=float(T_iso),
            init_mode="uninitialized",
            print_output=False,
            base_user_parameters=MIC_THERM_USER_PARAMETERS,
        )

        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] < 7:
            raise ValueError(
                f"MicTherm returned shape {values.shape} at T={T_iso}; "
                "at least seven VLE columns are required."
            )

        point_count = values.shape[0]
        results["T"].append(np.full(point_count, T_iso, dtype=float))
        results["VLE_x_l"].append(values[:, 1])
        results["p_VLE"].append(values[:, 2])
        results["VLE_y_v"].append(values[:, 3])
        results["VLE_rho_l"].append(values[:, 5])
        results["VLE_rho_v"].append(values[:, 6])

    return {name: np.concatenate(chunks) for name, chunks in results.items()}


def main():
    output_dir = PROJECT_ROOT / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "VLE_HFO-1234yf-CO2_temperature_scan.npz"

    print("Select VLE data source:")
    print(" 1) Calculate the VLE temperature scan again")
    print(" 2) Load the previously saved VLE scan")
    choice = input("Choose 1 or 2 [2]: ").strip() or "2"

    if choice == "1":
        results = calculate_vle_scan()
        np.savez(output_file, **results)
        print(f"Saved {results['T'].size} VLE points to {output_file}")
    elif choice == "2":
        if not output_file.exists():
            raise FileNotFoundError(
                f"{output_file} does not exist. Choose option 1 to create it."
            )
        with np.load(output_file) as data:
            missing = set(RESULT_NAMES).difference(data.files)
            if missing:
                raise ValueError(
                    f"The saved VLE scan is missing {sorted(missing)}. "
                    "Choose option 1 to calculate it again."
                )
            results = {name: np.asarray(data[name]) for name in RESULT_NAMES}
        print(
            f"Loaded {results['T'].size} VLE points from {output_file}. "
            "Choose option 1 if you want to calculate another grid."
        )
    else:
        raise ValueError("Choice must be 1 or 2")

    x_target = ask_float("Intersection plane x_CO2", 0.5)
    T_target = ask_float("Intersection plane temperature T", 283.0)
    intersection_temperatures = (T_target, 300.0)
    if not 0.0 <= x_target <= 1.0:
        raise ValueError("x_CO2 must be between 0 and 1")

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    density_min = min(
        np.min(results["VLE_rho_l"]), np.min(results["VLE_rho_v"])
    )
    density_max = max(
        np.max(results["VLE_rho_l"]), np.max(results["VLE_rho_v"])
    )
    density_norm = Normalize(vmin=density_min, vmax=density_max)
    density_cmap = "coolwarm"
    def add_density_lines(composition, pressure, density, cmap_name, norm):
        segments = []
        segment_density = []

        for T_iso in np.unique(results["T"]):
            mask = np.isclose(results["T"], T_iso)
            order = np.argsort(composition[mask])
            curve = np.column_stack(
                (
                    composition[mask][order],
                    pressure[mask][order],
                    results["T"][mask][order],
                )
            )
            curve_density = density[mask][order]
            if curve.shape[0] < 2:
                continue
            segments.extend(np.stack((curve[:-1], curve[1:]), axis=1))
            segment_density.extend(
                0.5 * (curve_density[:-1] + curve_density[1:])
            )

        lines = Line3DCollection(
            segments,
            cmap=cmap_name,
            norm=norm,
            linewidths=1.4,
            alpha=0.9,
        )
        lines.set_array(np.asarray(segment_density))
        ax.add_collection3d(lines)

    add_density_lines(
        results["VLE_x_l"],
        results["p_VLE"],
        results["VLE_rho_l"],
        density_cmap,
        density_norm,
    )
    add_density_lines(
        results["VLE_y_v"],
        results["p_VLE"],
        results["VLE_rho_v"],
        density_cmap,
        density_norm,
    )
    ax.auto_scale_xyz(
        np.concatenate((results["VLE_x_l"], results["VLE_y_v"])),
        np.concatenate((results["p_VLE"], results["p_VLE"])),
        np.concatenate((results["T"], results["T"])),
    )

    x_min = min(np.min(results["VLE_x_l"]), np.min(results["VLE_y_v"]))
    x_max = max(np.max(results["VLE_x_l"]), np.max(results["VLE_y_v"]))
    T_min = np.min(results["T"])
    T_max = np.max(results["T"])
    p_min = np.min(results["p_VLE"])
    p_max = np.max(results["p_VLE"])

    # The composition plane intersects both isotherm planes along pressure lines.
    plane_p, plane_T = np.meshgrid(
        np.linspace(p_min, p_max, 2), np.linspace(T_min, T_max, 2)
    )
    ax.plot_surface(
        np.full_like(plane_p, x_target),
        plane_p,
        plane_T,
        color="gray",
        alpha=0.16,
        linewidth=0,
        shade=False,
    )

    def interpolate_branch(composition, density, target_temperature):
        coordinates = np.column_stack((composition, results["T"]))
        _, unique_indices = np.unique(coordinates, axis=0, return_index=True)
        triangulation = Triangulation(
            composition[unique_indices], results["T"][unique_indices]
        )
        pressure_interpolator = LinearTriInterpolator(
            triangulation, results["p_VLE"][unique_indices]
        )
        density_interpolator = LinearTriInterpolator(
            triangulation, density[unique_indices]
        )
        pressure = pressure_interpolator(x_target, target_temperature)
        rho = density_interpolator(x_target, target_temperature)
        if np.ma.is_masked(pressure) or np.ma.is_masked(rho):
            return None
        return float(pressure), float(rho)

    for temperature_index, selected_T in enumerate(intersection_temperatures):
        plane_x, plane_p = np.meshgrid(
            np.linspace(x_min, x_max, 2), np.linspace(p_min, p_max, 2)
        )
        ax.plot_surface(
            plane_x,
            plane_p,
            np.full_like(plane_x, selected_T),
            color="gray",
            alpha=0.16,
            linewidth=0,
            shade=False,
        )
        ax.plot(
            [x_target, x_target],
            [p_min, p_max],
            [selected_T, selected_T],
            color="black",
            linewidth=1.1,
            linestyle="--",
            label="Plane intersection" if temperature_index == 0 else None,
        )

        liquid_state = interpolate_branch(
            results["VLE_x_l"], results["VLE_rho_l"], selected_T
        )
        vapor_state = interpolate_branch(
            results["VLE_y_v"], results["VLE_rho_v"], selected_T
        )
        phase_states = (
            ("liquid", liquid_state, "darkred", "o"),
            ("vapor", vapor_state, "navy", "^"),
        )
        rho_text = [rf"$T={selected_T:g}\,\mathrm{{K}}$"]
        for phase, state, color, marker in phase_states:
            if state is None:
                rho_text.append(
                    rf"$\rho_{{\mathrm{{{phase[0]}}}}}$ = "
                    r"$\mathrm{outside\ range}$"
                )
                print(
                    f"No {phase} VLE intersection at x_CO2={x_target}, "
                    f"T={selected_T}; the point is outside that data surface."
                )
                continue
            pressure, rho = state
            ax.scatter(
                [x_target],
                [pressure],
                [selected_T],
                color=color,
                marker=marker,
                s=55,
                depthshade=False,
                label=(
                    f"{phase.capitalize()} intersection"
                    if temperature_index == 0
                    else None
                ),
            )
            rho_text.append(
                rf"$\rho_{{\mathrm{{{phase[0]}}}}}={rho:.5g}"
                r"\,\mathrm{mol\,L^{-1}}$"
            )
            print(
                f"{phase.capitalize()} VLE intersection: "
                f"x_CO2={x_target}, T={selected_T}, "
                f"p_VLE={pressure}, rho={rho}"
            )

        text_x = 0.02
        text_y = 0.98 - 0.18 * temperature_index
        ax.text2D(
            text_x,
            text_y,
            "\n".join(rho_text),
            transform=ax.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
            color="black",
            fontsize=18,
            zorder=1000,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "gray",
                "alpha": 0.9,
            },
        )

    ax.set_xlabel(r"$x_{\mathrm{CO_2}}$ / $\mathrm{mol\,mol^{-1}}$")
    ax.set_ylabel(r"$p_{\mathrm{VLE}}$ / $\mathrm{MPa}$")
    ax.set_zlabel(r"$T$ / $\mathrm{K}$")
    ax.set_title("HFO-1234yf/CO2 VLE")
    density_colorbar = fig.colorbar(
        ScalarMappable(norm=density_norm, cmap=density_cmap),
        ax=ax,
        pad=0.08,
        shrink=0.72,
    )
    density_colorbar.set_label(
        r"$\mathrm{VLE\ density}\ "
        r"\rho_{\mathrm{v}}\rightarrow\rho_{\mathrm{l}}$"
        r" / $\mathrm{mol\,L^{-1}}$"
    )

    fig.tight_layout()

    # Separate pure-component saturation-pressure diagram. For the composition
    # coordinate returned in VLE_x_l/VLE_y_v, x=0 is pure CO2 and x=1 is pure
    # HFO-1234yf.
    def pure_component_curve(target_composition):
        curve_T = []
        curve_p = []
        for T_iso in np.unique(results["T"]):
            mask = np.isclose(results["T"], T_iso)
            endpoint_pressures = []
            for composition in (results["VLE_x_l"], results["VLE_y_v"]):
                branch_x = composition[mask]
                branch_p = results["p_VLE"][mask]
                order = np.argsort(branch_x)
                branch_x, unique_indices = np.unique(
                    branch_x[order], return_index=True
                )
                branch_p = branch_p[order][unique_indices]
                if (
                    branch_x.size >= 2
                    and branch_x[0] <= target_composition <= branch_x[-1]
                ):
                    endpoint_pressures.append(
                        np.interp(target_composition, branch_x, branch_p)
                    )
            if endpoint_pressures:
                curve_T.append(T_iso)
                curve_p.append(np.mean(endpoint_pressures))
        return np.asarray(curve_T), np.asarray(curve_p)

    def extrapolate_saturation_pressure(curve_T, curve_p, critical_temperature):
        """Short log-pressure extrapolation based on the last VLE points."""
        valid = np.isfinite(curve_T) & np.isfinite(curve_p) & (curve_p > 0.0)
        fit_T = curve_T[valid]
        fit_p = curve_p[valid]
        if fit_T.size < 2 or fit_T[-1] >= critical_temperature:
            return np.array([]), np.array([])

        fit_count = min(5, fit_T.size)
        polynomial_degree = min(2, fit_count - 1)
        coefficients = np.polyfit(
            fit_T[-fit_count:],
            np.log(fit_p[-fit_count:]),
            polynomial_degree,
        )
        available_steps = np.diff(np.unique(fit_T))
        temperature_step = (
            np.median(available_steps[available_steps > 0.0])
            if np.any(available_steps > 0.0)
            else critical_temperature - fit_T[-1]
        )
        extrapolated_T = np.arange(
            fit_T[-1] + temperature_step,
            critical_temperature,
            temperature_step,
        )
        extrapolated_T = np.append(extrapolated_T, critical_temperature)
        extrapolated_p = np.exp(np.polyval(coefficients, extrapolated_T))
        return extrapolated_T, extrapolated_p

    pure_curves = (
        (0.0, r"$\mathrm{CO_2}$", "--", "s", 304.13),
        (1.0, "HFO-1234yf", "-", "o", 367.85),
    )
    fig_pure, ax_pure = plt.subplots(figsize=(8, 6))
    pure_pressure_max = None
    for pure_x, substance, linestyle, marker, critical_temperature in pure_curves:
        pure_T, pure_p = pure_component_curve(pure_x)
        if pure_T.size == 0:
            print(
                f"No pure-component endpoint data found at x_CO2={pure_x} "
                f"for {substance}."
            )
            continue
        below_critical = pure_T <= critical_temperature
        pure_T = pure_T[below_critical]
        pure_p = pure_p[below_critical]
        if pure_T.size < 2:
            print(
                f"Not enough {substance} endpoint data below "
                f"T_crit={critical_temperature:g} K."
            )
            continue
        extrapolated_T, extrapolated_p = extrapolate_saturation_pressure(
            pure_T, pure_p, critical_temperature
        )
        if extrapolated_T.size:
            pure_T = np.concatenate((pure_T, extrapolated_T))
            pure_p = np.concatenate((pure_p, extrapolated_p))
        curve_pressure_max = np.max(pure_p)
        pure_pressure_max = (
            curve_pressure_max
            if pure_pressure_max is None
            else max(pure_pressure_max, curve_pressure_max)
        )

        ax_pure.plot(
            pure_T,
            pure_p,
            color="black",
            linestyle=linestyle,
            linewidth=1.6,
            marker=marker,
            markersize=3.5,
            label=rf"{substance}: $p_\mathrm{{sat}}(T)$",
        )
        ax.plot(
            np.full_like(pure_T, pure_x),
            pure_p,
            pure_T,
            color="black",
            linestyle=linestyle,
            linewidth=2.0,
            label=rf"{substance}: pure-component line",
        )
        ax_pure.scatter(
            [pure_T[-1]],
            [pure_p[-1]],
            color="black",
            marker="X",
            s=55,
            zorder=5,
        )
        ax_pure.annotate(
            rf"$T_\mathrm{{crit}}={pure_T[-1]:g}\,\mathrm{{K}}$",
            xy=(pure_T[-1], pure_p[-1]),
            xytext=(-8, 8),
            textcoords="offset points",
            horizontalalignment="right",
            fontsize=12,
        )
        ax.scatter(
            [pure_x],
            [pure_p[-1]],
            [pure_T[-1]],
            color="black",
            marker="X",
            s=55,
            depthshade=False,
        )

    ax_pure.set_xlabel(r"$T$ / $\mathrm{K}$")
    ax_pure.set_ylabel(r"$p_{\mathrm{sat}}$ / $\mathrm{MPa}$")
    ax_pure.set_title("Pure-component saturation-pressure curves")
    ax_pure.grid(True, color="0.85", linewidth=0.7)
    if pure_pressure_max is not None:
        current_bottom, _ = ax_pure.get_ylim()
        ax_pure.set_ylim(current_bottom, 1.15 * pure_pressure_max)
    ax_pure.legend()
    fig_pure.tight_layout()
    ax.legend(loc="best")
    fig.tight_layout()

    # A separate two-dimensional p-x diagram for each selected isotherm.
    def plot_px_isotherm(selected_T):
        fig_px, ax_px = plt.subplots(figsize=(8, 6))
        branch_definitions = (
            (
                "liquid",
                results["VLE_x_l"],
                results["VLE_rho_l"],
                "solid",
                r"liquid branch $x_{\mathrm{CO_2}}$",
            ),
            (
                "vapor",
                results["VLE_y_v"],
                results["VLE_rho_v"],
                "dashed",
                r"vapor branch $y_{\mathrm{CO_2}}$",
            ),
        )
        plotted_branches = []
        branch_slices = {}

        for phase, composition, density, linestyle, label in branch_definitions:
            coordinates = np.column_stack((composition, results["T"]))
            _, unique_indices = np.unique(
                coordinates, axis=0, return_index=True
            )
            triangulation = Triangulation(
                composition[unique_indices], results["T"][unique_indices]
            )
            pressure_interpolator = LinearTriInterpolator(
                triangulation, results["p_VLE"][unique_indices]
            )
            density_interpolator = LinearTriInterpolator(
                triangulation, density[unique_indices]
            )

            composition_axis = np.linspace(
                np.min(composition), np.max(composition), 500
            )
            temperature_axis = np.full_like(composition_axis, selected_T)
            pressure_slice = pressure_interpolator(
                composition_axis, temperature_axis
            )
            density_slice = density_interpolator(
                composition_axis, temperature_axis
            )
            valid = ~(
                np.ma.getmaskarray(pressure_slice)
                | np.ma.getmaskarray(density_slice)
            )
            slice_x = composition_axis[valid]
            slice_p = np.asarray(pressure_slice)[valid]
            slice_rho = np.asarray(density_slice)[valid]
            if slice_x.size < 2:
                print(f"No p-x curve available at T={selected_T:g} K.")
                continue

            points = np.column_stack((slice_x, slice_p))
            segments = np.stack((points[:-1], points[1:]), axis=1)
            segment_density = 0.5 * (slice_rho[:-1] + slice_rho[1:])
            collection = LineCollection(
                segments,
                cmap=density_cmap,
                norm=density_norm,
                linewidths=2.0,
                linestyles=linestyle,
            )
            collection.set_array(segment_density)
            ax_px.add_collection(collection)
            plotted_branches.append((linestyle, label))
            branch_slices[phase] = (slice_x, slice_p, slice_rho)

        ax_px.autoscale()

        # Bubble-point construction for the selected bulk composition:
        # vertical from z to the liquid branch, then a horizontal tie-line to
        # the equilibrium vapor composition.
        if "liquid" in branch_slices and "vapor" in branch_slices:
            liquid_x, liquid_p, _ = branch_slices["liquid"]
            vapor_y, vapor_p, _ = branch_slices["vapor"]
            if liquid_x[0] <= x_target <= liquid_x[-1]:
                bubble_pressure = np.interp(x_target, liquid_x, liquid_p)
                pressure_difference = vapor_p - bubble_pressure
                crossing_indices = np.flatnonzero(
                    pressure_difference[:-1] * pressure_difference[1:] <= 0.0
                )
                vapor_candidates = []
                for crossing_index in crossing_indices:
                    p0 = vapor_p[crossing_index]
                    p1 = vapor_p[crossing_index + 1]
                    y0 = vapor_y[crossing_index]
                    y1 = vapor_y[crossing_index + 1]
                    if np.isclose(p0, p1):
                        vapor_candidates.append(0.5 * (y0 + y1))
                    else:
                        weight = (bubble_pressure - p0) / (p1 - p0)
                        vapor_candidates.append(y0 + weight * (y1 - y0))

                if vapor_candidates:
                    # If numerical interpolation produces several crossings,
                    # use the one furthest from x to show the phase split.
                    equilibrium_y = max(
                        vapor_candidates,
                        key=lambda value: abs(value - x_target),
                    )
                    pressure_floor = min(
                        np.min(liquid_p), np.min(vapor_p)
                    )
                    ax_px.plot(
                        [x_target, x_target],
                        [pressure_floor, bubble_pressure],
                        color="black",
                        linestyle=":",
                        linewidth=1.4,
                    )
                    ax_px.plot(
                        [x_target, equilibrium_y],
                        [bubble_pressure, bubble_pressure],
                        color="black",
                        linestyle="-",
                        linewidth=1.8,
                        label="tie-line",
                    )
                    ax_px.scatter(
                        [x_target, equilibrium_y],
                        [bubble_pressure, bubble_pressure],
                        color="black",
                        marker="o",
                        s=30,
                        zorder=5,
                    )
                    ax_px.annotate(
                        rf"$x={x_target:.3g}$",
                        xy=(x_target, bubble_pressure),
                        xytext=(5, 8),
                        textcoords="offset points",
                        fontsize=11,
                    )
                    ax_px.annotate(
                        rf"$y={equilibrium_y:.3g}$",
                        xy=(equilibrium_y, bubble_pressure),
                        xytext=(5, -16),
                        textcoords="offset points",
                        fontsize=11,
                    )
                    ax_px.annotate(
                        rf"$p={bubble_pressure:.5g}\,\mathrm{{MPa}}$",
                        xy=(
                            0.5 * (x_target + equilibrium_y),
                            bubble_pressure,
                        ),
                        xytext=(0, 8),
                        textcoords="offset points",
                        horizontalalignment="center",
                        fontsize=11,
                        bbox={
                            "boxstyle": "round,pad=0.2",
                            "facecolor": "white",
                            "edgecolor": "black",
                            "alpha": 0.9,
                        },
                    )
                    print(
                        f"Tie-line at T={selected_T:g} K: "
                        f"x={x_target:.6g}, y={equilibrium_y:.6g}, "
                        f"p={bubble_pressure:.6g} MPa"
                    )
                    plotted_branches.append(("solid", "tie-line"))
                else:
                    print(
                        f"No vapor-branch crossing at the bubble pressure "
                        f"for T={selected_T:g} K and x={x_target:g}."
                    )
            else:
                print(
                    f"x={x_target:g} is outside the liquid branch at "
                    f"T={selected_T:g} K."
                )

        ax_px.set_xlabel(
            r"$x_{\mathrm{CO_2}}$ / $\mathrm{mol\,mol^{-1}}$"
        )
        ax_px.set_ylabel(r"$p_{\mathrm{VLE}}$ / $\mathrm{MPa}$")
        ax_px.set_title(
            rf"$p$-$x$ diagram at $T={selected_T:g}\,\mathrm{{K}}$"
        )
        ax_px.grid(True, color="0.85", linewidth=0.7)
        if plotted_branches:
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=2.0,
                    linestyle=linestyle,
                    label=label,
                )
                for linestyle, label in plotted_branches
            ]
            ax_px.legend(handles=legend_handles)
        px_colorbar = fig_px.colorbar(
            ScalarMappable(norm=density_norm, cmap=density_cmap),
            ax=ax_px,
            pad=0.03,
        )
        px_colorbar.set_label(r"$\rho$ / $\mathrm{mol\,L^{-1}}$")
        fig_px.tight_layout()

    for selected_T in intersection_temperatures:
        plot_px_isotherm(selected_T)

    plt.show()


if __name__ == "__main__":
    main()
