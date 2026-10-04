"""
Smoke test: 2D VdW droplet with the cached MicTherm grid (temp/temp_mictherm_grids_VdW.npz).

Run from the repository root:
    py -3.11 tests/smoke_vdw_droplet.py

Non-interactive, no MicTherm runtime needed, writes only to output/smoke/.
Exit code 0 = PASS, 1 = FAIL.
"""

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # no plot windows

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VDW_SCRIPT = PROJECT_ROOT / "MicLBM_examples" / "multiphase" / "MRT" / "grid_generation" / "droplet_2d_2D_Mic_VdW.py"
OUTPUT_DIR = PROJECT_ROOT / "output" / "smoke"

N_STEPS = 200
TR = 0.8

# Acceptance limits (see memories.md for the reference run).
MAX_SPURIOUS_CURRENTS = 0.1
MAX_ABS_ERROR_RHO_L_PERCENT = 10.0
MAX_ABS_ERROR_RHO_G_PERCENT = 5.0


def load_vdw_module():
    spec = importlib.util.spec_from_file_location("droplet_2d_2D_Mic_VdW", VDW_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(metrics):
    failures = []
    if metrics is None:
        return ["output_data was never called"]
    if metrics["timestep"] < N_STEPS:
        failures.append(f"last output at timestep {metrics['timestep']} < {N_STEPS}")
    if not metrics["all_finite"]:
        failures.append("NaN/inf in rho, p or u")
    if not metrics["rho_l_pred"] > metrics["rho_g_pred"]:
        failures.append("droplet vanished (rho_l <= rho_g)")
    if metrics["spurious_currents"] > MAX_SPURIOUS_CURRENTS:
        failures.append(f"spurious currents {metrics['spurious_currents']:.3g} > {MAX_SPURIOUS_CURRENTS}")
    if abs(metrics["error_rho_l_percent"]) > MAX_ABS_ERROR_RHO_L_PERCENT:
        failures.append(f"|error rho_l| {abs(metrics['error_rho_l_percent']):.3g}% > {MAX_ABS_ERROR_RHO_L_PERCENT}%")
    if abs(metrics["error_rho_g_percent"]) > MAX_ABS_ERROR_RHO_G_PERCENT:
        failures.append(f"|error rho_g| {abs(metrics['error_rho_g_percent']):.3g}% > {MAX_ABS_ERROR_RHO_G_PERCENT}%")
    return failures


def main():
    vdw = load_vdw_module()
    grid = vdw.load_grid()
    sim = vdw.build_simulation(grid, Tr=TR, output_dir=str(OUTPUT_DIR))
    sim.run(N_STEPS)

    metrics = sim.last_metrics
    failures = check(metrics)
    print("\nSmoke test metrics:", metrics)
    if failures:
        print("SMOKE TEST: FAIL")
        for failure in failures:
            print(" -", failure)
        return 1
    print("SMOKE TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
