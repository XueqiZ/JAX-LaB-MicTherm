# MicTherm Grid Generation Examples

This folder contains the example scripts that calculate MicTherm thermodynamic property grids or reusable VLE archives. The cached files are written to the repository-level `temp/` directory, not into this folder.

These scripts are also complete droplet examples. Their grid-generation mode prepares the EOS tables used later by the LBM simulation.

## Publication-relevant workflows

These are the three grid generators relevant to the main publication workflow:

| Script | Model or fluid | Reusable cache | Main purpose |
| --- | --- | --- | --- |
| `droplet_2d_2D_Mic_CO2.py` | CO2, M-SAFT-VR-Mie | `temp_mictherm_grids_CO2.npz` | CO2 critical point, VLE curve, pressure, viscosity, and surface-tension grids (`SI_reduced`, reduced values). |
| `droplet_2d_2D_Mic_PeTS.py` | Single-component PeTS | `temp_mictherm_grids_PeTS.npz` | PeTS EOS grid used by the publication test cases and PeTS droplet example. |
| `droplet_2d_2D_Mic_VdW.py` | VdW reference model | `temp_mictherm_grids_VdW.npz` (SI), `temp_mictherm_grids_VdW_SI_reduced.npz` | VdW property grid; asks for `SI` (default, ρ and p equal the lattice VdW values) or `SI_reduced`. Used by `tests/smoke_vdw_droplet.py`. |

Use these three first when reproducing the publication simulations or preparing their EOS data. They cover the CO2, PeTS, and VdW cases used as the main reference workflows.

Each main script asks whether to use an existing cache or generate a new one. Choose the generation option only when the MicTherm runtime is available. The normal workflow is:

1. Run the relevant script in generation mode once.
2. The script writes its `.npz` archive to `temp/`.
3. Run it again in cache mode, or run a publication test case that loads the archive.

## Side project (not for the publication)

`droplet_2d_2D_Mic_HFO-1234yf.py`, `droplet_2d_2D_Mic_HFO-1234yf-CO2.py` and `calculate_VLE_HFO-1234yf-CO2.py`
belong to a separate HFO-1234yf / CO2 project. They are not part of the publication workflow, and their caches are
not kept in `temp/`.

## Checking a cache

Each main script plots the VLE branches and the critical point on the p(ρ, T) surface and prints
max |p_l − p_g| / p_c (interpolation error of the grid). For cached grids without running a script:

```powershell
python MicLBM_src/grid_plot.py            # VdW_SI_reduced, CO2, PeTS
python MicLBM_src/grid_plot.py VdW        # VdW SI cache
```

## Legacy VdW generators

These files are retained for older VdW tests and write the simpler cache `temp/temp_mictherm_grids.npz`. They are not needed when using the publication VdW generator above:

- `droplet_2d_2D_Mic_VdW_legacy.py`
- `droplet_2d_2D_Mic_VdW_Trinangle_Profil.py`

They save only `T_grid`, `rho_grid`, and `p_grid`, so they are not interchangeable with the newer named caches above.

## Related in-memory grid examples

The `Variation of droplet_2d/` directory still contains scripts that call `mictherm_grid()` but do not save a reusable `.npz` archive:

- `droplet_2d_1D_Mic_CO2.py`
- `droplet_2d_1D_Mic_PR_Vali.py`
- `droplet_2d_1D_Mic_VdW.py`
- `droplet_2d_1D_Mic_VdW_Vali.py`

Use these for one-off validation or profile calculations rather than for building a shared cache.

## Output locations

- EOS caches: repository `temp/` directory.
- Pressure-grid plots and simulation output: repository `output/` or the script's configured output path.
- Publication and test-case consumers: `Publication test case/`.

The scripts add the repository root to `sys.path` using their current location, so they remain runnable after being organized in this folder.
