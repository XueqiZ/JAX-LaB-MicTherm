[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# JAX-LaB-MicTherm

**Lattice Boltzmann multiphase simulations with molecular-based equations of state.**

JAX-LaB-MicTherm couples the differentiable lattice Boltzmann library [JAX-LaB](https://github.com/piyush-ppradhan/JAX-LaB)
(Shan-Chen pseudopotential, MRT collision) with **MicTherm**, a thermodynamics package (MATLAB-compiled) that provides
equations of state from simple cubic models up to molecular-based ones:

- **van der Waals (VdW)** — analytic reference case
- **PeTS** — perturbed truncated and shifted Lennard-Jones fluid
- **M-SAFT-VR-Mie** — real fluids, e.g. **CO2**

Instead of a hard-coded analytic EOS, the LBM reads pressure, viscosity and surface tension from MicTherm property
grids p(ρ, T), η(ρ, T), γ(T). The grids are computed once, cached, and interpolated inside the simulation, so
non-isothermal cases (spatially varying temperature field) work with any MicTherm fluid.

> A publication describing the coupling is in preparation.

## Workflow

```
 MicTherm (MATLAB Runtime)            cache                       LBM (JAX)
 ─────────────────────────      ───────────────────      ─────────────────────────────
 critical point  Tc, ρc, pc  ─┐
 VLE curve       ρl(T), ρg(T) ─┼─►  temp/temp_mictherm_  ─►  MicTherm EOS: interpolate
 property grid   p, η, γ      ─┘    grids_<fluid>.npz        p(ρ, T) on the local T field
                                                             Maxwell densities → initial state
```

1. **Generate the grid** (needs the MicTherm runtime, seconds to minutes): critical point, VLE curve and the
   property grid on T = Trmin…1 · Tc, ρ = 0…ρl(Trmin), saved as `.npz`.
2. **Run the LBM** from the cache (no MATLAB needed): `MicLBM_src.eos.MicTherm` interpolates p(ρ, T) at every node,
   `MicLBM_src.Mic_multiphase.MultiphaseMRTTvar` adds the temperature field to the multiphase MRT model.
3. **Check consistency** of a cache: VLE branches and critical point plotted on the p(ρ, T) surface.

## Fluids and units

All quantities in one cache (critical point, VLE, grids) share one unit system. Unit systems may differ between fluids.

| Fluid | EOS | MicTherm `units` | Cache | Notes |
|---|---|---|---|---|
| VdW | `vanderWaals` | `SI` (default) or `SI_reduced` | `temp_mictherm_grids_VdW.npz` / `..._VdW_SI_reduced.npz` | SI: a, b = lattice values × 1000 → ρ [mol/L] and p [MPa] equal the lattice VdW values (ρc = 3.5, pc = 0.75) |
| CO2 | `M_SAFT_VR_MIE` | `SI_reduced` | `temp_mictherm_grids_CO2.npz` | reduced grid |
| PeTS | PeTS | `REDUCED` | `temp_mictherm_grids_PeTS.npz` | reduced LJ units |

Things worth knowing:

- **`SI_reduced`** takes SI parameters and returns reduced values, but the state inputs (T, ρ) of
  `userproperties` must still be given in SI (K, mol/L). The grid generators convert the reduced grid to SI only for
  that call. The reduction is a fixed scaling (ε/k = 200 K, σ = 3.5 Å, i.e. T* = T / 200 K, ρ* = ρ / 38.73, p* = p / 64.40).
- **VdW in SI has Tc = 68.73 K, not 4/7.** For VdW, ρc = 1/(3b), pc = a/(27b²), Tc = 8a/(27bR); only Tc contains R.
  With R = 8.314 J/(mol K) (instead of the lattice R = 1) only two of the three lattice values can be matched; ρ and p
  are matched because the LBM uses them directly. T always enters as Tr = T/Tc, so the simulation is unchanged.
- The LBM parameters (k, A) in the VdW example are tuned for the SI grid; they are untested with `SI_reduced`.
- Grid resolution matters: coarse grids (e.g. ΔT = 0.1 Tc) give visible interpolation errors on the steep liquid
  branch. The defaults (Tstep = 0.01, rhostep = 0.03) keep the VLE pressure mismatch at a few percent of pc.

## Installation

Tested on Windows with Python 3.11 (MicTherm supports Python 3.9–3.12).

1. **Python dependencies**
   ```bash
   python -m pip install -r requirements.txt scipy pandas
   ```
   JAX: see the [JAX installation guide](https://github.com/google/jax#installation) for GPU/TPU builds.
2. **MATLAB Runtime R2025a** (free) in its default location
   (`C:\Program Files\MATLAB\MATLAB Runtime\R2025a`); the path is set in `MicLBM_src/MicTherm_API.py`.
   Only needed to generate new grids — running the LBM from cached grids does not need it.
3. **MicTherm package** — MathWorks-generated, in `MicLBM_src/MicTherm/` with its `MicLBM_src/setup.py`
   (install it following the MathWorks instructions for compiled Python packages).

## Quick start

Run everything from the repository root; the scripts add the root to `sys.path` themselves.

```bash
# Smoke test: VdW droplet from the cached grid, 200 steps, ~10 s, writes to output/smoke/, prints PASS/FAIL
python tests/smoke_vdw_droplet.py

# Consistency plot of cached grids (VLE + critical point on the p surface); read-only
python MicLBM_src/grid_plot.py                 # VdW_SI_reduced, CO2, PeTS
python MicLBM_src/grid_plot.py VdW             # VdW SI cache

# Droplet example: asks for grid mode (1 = cache, 2 = regenerate with MicTherm), units and Tr
python MicLBM_examples/multiphase/MRT/grid_generation/droplet_2d_2D_Mic_VdW.py
```

The same pattern exists for `droplet_2d_2D_Mic_CO2.py` and `droplet_2d_2D_Mic_PeTS.py`
(see [grid_generation/README.md](MicLBM_examples/multiphase/MRT/grid_generation/README.md)).

## Repository structure

| Path | Content |
|---|---|
| `src/` | Upstream JAX-LaB core (lattices, boundary conditions, collision models, multiphase). Unmodified. |
| `MicLBM_src/` | Coupling code: `eos.py` (`MicTherm` EOS, `interpolate_mictherm_properties`), `Mic_multiphase.py` (`MultiphaseMRTTvar`, temperature field), `MicTherm_API.py` + `run_mictherm.py` (MATLAB bridge), `grid_plot.py` (cache check), `MicTherm/` (generated package) |
| `MicLBM_examples/multiphase/MRT/grid_generation/` | Grid generators and droplet cases for VdW, CO2, PeTS |
| `MicLBM_examples/multiphase/MRT/Publication test case/` | Simulations for the publication (phase separation, buoyant bubbles, channel flow, capillary rise) |
| `MicLBM_examples/kA_Opt/`, `.../MRT/optimize_kA.py` | Bayesian optimization of the Shan-Chen parameters k, A |
| `tests/` | Smoke test |
| `temp/` | Cached MicTherm grids `temp_mictherm_grids_<fluid>.npz` |
| `output/` | Simulation results (VTK, CSV, images); not tracked |
| `examples/`, `examples_SDU/`, `data/`, `docs/` | Upstream JAX-LaB examples, validation data and docs |

## Credits and license

Built on [JAX-LaB](https://github.com/piyush-ppradhan/JAX-LaB) by P. Pradhan et al., itself an extension of
[XLB](https://github.com/Autodesk/XLB). For the LBM features (collision models, boundary conditions, wetting,
multi-GPU, differentiability) see the JAX-LaB repository and its paper in
[JAMES (2025)](https://doi.org/10.1029/2025MS005313).

Licensed under the Apache License 2.0, as the upstream projects.
